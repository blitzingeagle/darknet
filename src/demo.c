#include "network.h"
#include "detection_layer.h"
#include "region_layer.h"
#include "cost_layer.h"
#include "utils.h"
#include "parser.h"
#include "box.h"
#include "image.h"
#include "demo.h"
#include <sys/time.h>
#include "json.h"
#define DEMO 1

#ifdef OPENCV

static char **demo_names;
static image **demo_alphabet;
static int demo_classes;

static float **probs;
static box *boxes;
static network *net;
static image buff [3];
static image buff_letter[3];
static int buff_index = 0;
static CvCapture * cap;
static IplImage  * ipl;
static float fps = 0;
static float demo_thresh = 0;
static float demo_hier = .5;
static int running = 0;

static int demo_frame = 3;
static int demo_detections = 0;
static float **predictions;
static int demo_index = 0;
static int demo_done = 0;
static float *avg;
double demo_time;
const char *whitelist_name = "data/whitelist.name";
char **whitelist = NULL;
int listsize = 0;
int total_videoframes = 0;
char tag_results[4096] = {0};

void output_result(char *indata)
{
    json_object *jroot = json_object_new_object();
    json_object *data = json_object_new_object();
    json_object_object_add(data,"outputfile",json_object_new_string(indata));
    json_object_object_add(jroot,"data",data);
    json_object_object_add(jroot,"type",json_object_new_string("RESULT"));
    
    printf("%s\n",json_object_to_json_string_ext(jroot,JSON_C_TO_STRING_NOSLASHESCAPE));
    //json_object_put(data);
    json_object_put(jroot);
}

void write_json_file(FILE *file, char* image_name, char* tag)
{
    json_object *jroot = json_object_new_object();
    json_object_object_add(jroot,"filename",json_object_new_string(image_name));
    json_object *tag_array = json_object_new_array();
    
    char *pch;
    char tag_tmp[4096];
    strcpy(tag_tmp,tag);
    pch = strtok(tag_tmp,"|");
    while(pch != NULL){
        json_object *split_tag = json_tokener_parse(pch);
        //printf("a30323: %s\n",json_object_to_json_string_ext(split_tag,JSON_C_TO_STRING_NOSLASHESCAPE));
        json_object_array_add(tag_array,split_tag);
        pch = strtok(NULL,"|");
    }
    
    //json_object_object_add(jroot,"tag",json_object_new_string(tag));
    json_object_object_add(jroot,"tag",tag_array);
    char buffer[4096] = {0};
    sprintf(buffer,"%s\n",json_object_to_json_string_ext(jroot,JSON_C_TO_STRING_NOSLASHESCAPE));
    fwrite(buffer,sizeof(char),strlen(buffer)*sizeof(char),file);
    //printf("%s",buffer);
    json_object_put(jroot);
}

void update_progress(int progress)
{
    json_object *jroot = json_object_new_object();
    json_object *data = json_object_new_object();
    json_object_object_add(data,"progress",json_object_new_int(progress));
    json_object_object_add(jroot,"data",data);
    json_object_object_add(jroot,"type",json_object_new_string("UPDATESTATUS"));

    printf("%s\n",json_object_to_json_string_ext(jroot,JSON_C_TO_STRING_NOSLASHESCAPE));
    //json_object_put(data);
    json_object_put(jroot);
}

void save_image_video(CvVideoWriter *writer, image img){
    image copy = copy_image(img);
    if(img.c == 3) rgbgr_image(copy);
    IplImage *disp = cvCreateImage(cvSize(img.w,img.h), IPL_DEPTH_8U, img.c);
    int step = disp->widthStep;
    for(int y = 0; y < img.h; ++y){
        for(int x = 0; x < img.w; ++x){
            for(int k= 0; k < img.c; ++k){
                disp->imageData[y*step + x*img.c + k] = (unsigned char)(copy.data[k*copy.h*copy.w + y*copy.w + x]*255);
            }
        }
    }
    cvWriteFrame(writer,disp);
    cvReleaseImage(&disp);
    free_image(copy);
}

void *detect_in_thread(void *ptr)
{
    //printf("enter detect\n");
    running = 1;
    float nms = .4;

    layer l = net->layers[net->n-1];
    float *X = buff_letter[(buff_index+2)%3].data;
    float *prediction = network_predict(net, X);

    memcpy(predictions[demo_index], prediction, l.outputs*sizeof(float));
    mean_arrays(predictions, demo_frame, l.outputs, avg);
    l.output = avg;
    if(l.type == DETECTION){
        get_detection_boxes(l, 1, 1, demo_thresh, probs, boxes, 0);
    } else if (l.type == REGION){
        get_region_boxes(l, buff[0].w, buff[0].h, net->w, net->h, demo_thresh, probs, boxes, 0, 0, 0, demo_hier, 1);
    } else {
        error("Last layer must produce detections\n");
    }
    if (nms > 0) do_nms_obj(boxes, probs, l.w*l.h*l.n, l.classes, nms);

    //printf("\033[2J");
    //printf("\033[1;1H");
    //printf("\nFPS:%.1f\n",fps);
    printf("Objects:\n");
    image display = buff[(buff_index+2) % 3];
    
    draw_detections_whitelist(display, demo_detections, demo_thresh, boxes, probs, 0, demo_names, demo_alphabet, demo_classes, whitelist, listsize, tag_results);
    //printf("detect_in_thread : %s\n", tag_results);
    demo_index = (demo_index + 1)%demo_frame;
    running = 0;
    return 0;
}

void *fetch_in_thread(void *ptr)
{
    //printf("enter fetch\n");
    int status = fill_image_from_stream(cap, buff[buff_index]);
    letterbox_image_into(buff[buff_index], net->w, net->h, buff_letter[buff_index]);
    if(status == 0) demo_done = 1;
    //printf("leave fetch\n");
    return 0;
}

void *display_in_thread(void *ptr)
{
    show_image_cv(buff[(buff_index + 1)%3], "Demo", ipl);
    int c = cvWaitKey(1);
    if (c != -1) c = c%256;
    if (c == 27) {
        demo_done = 1;
        return 0;
    } else if (c == 82) {
        demo_thresh += .02;
    } else if (c == 84) {
        demo_thresh -= .02;
        if(demo_thresh <= .02) demo_thresh = .02;
    } else if (c == 83) {
        demo_hier += .02;
    } else if (c == 81) {
        demo_hier -= .02;
        if(demo_hier <= .0) demo_hier = .0;
    }
    return 0;
}

void *display_loop(void *ptr)
{
    while(1){
        display_in_thread(0);
    }
}

void *detect_loop(void *ptr)
{
    while(1){
        detect_in_thread(0);
    }
}

void demo(char *cfgfile, char *weightfile, float thresh, int cam_index, const char *filename, char **names, int classes, int delay, char *prefix, int avg_frames, float hier, int w, int h, int frames, int fullscreen, const char *outputvideo)
{
    demo_frame = avg_frames;
    predictions = calloc(demo_frame, sizeof(float*));
    image **alphabet = load_alphabet();
    demo_names = names;
    demo_alphabet = alphabet;
    demo_classes = classes;
    demo_thresh = thresh;
    demo_hier = hier;
    //printf("Demo\n");
    net = load_network(cfgfile, weightfile, 0);
    set_batch_network(net, 1);
    pthread_t detect_thread;
    pthread_t fetch_thread;

    srand(2222222);

    if(filename){
        printf("video file: %s\n", filename);
        printf("output video file: %s\n", outputvideo);
        cap = cvCaptureFromFile(filename);
        total_videoframes = cvGetCaptureProperty(cap,CV_CAP_PROP_FRAME_COUNT);
        printf("video frames : %d\n",total_videoframes);
    }else{
        cap = cvCaptureFromCAM(cam_index);

        if(w){
            cvSetCaptureProperty(cap, CV_CAP_PROP_FRAME_WIDTH, w);
        }
        if(h){
            cvSetCaptureProperty(cap, CV_CAP_PROP_FRAME_HEIGHT, h);
        }
        if(frames){
            cvSetCaptureProperty(cap, CV_CAP_PROP_FPS, frames);
        }
    }

    if(!cap) error("Couldn't connect to webcam.\n");

    layer l = net->layers[net->n-1];
    demo_detections = l.n*l.w*l.h;

    avg = (float *) calloc(l.outputs, sizeof(float));
    for(int j = 0; j < demo_frame; ++j) predictions[j] = (float *) calloc(l.outputs, sizeof(float));

    boxes = (box *)calloc(l.w*l.h*l.n, sizeof(box));
    probs = (float **)calloc(l.w*l.h*l.n, sizeof(float *));
    for(int j = 0; j < l.w*l.h*l.n; ++j) probs[j] = (float *)calloc(l.classes+1, sizeof(float));

    buff[0] = get_image_from_stream(cap);
    buff[1] = copy_image(buff[0]);
    buff[2] = copy_image(buff[0]);
    buff_letter[0] = letterbox_image(buff[0], net->w, net->h);
    buff_letter[1] = letterbox_image(buff[0], net->w, net->h);
    buff_letter[2] = letterbox_image(buff[0], net->w, net->h);
    ipl = cvCreateImage(cvSize(buff[0].w,buff[0].h), IPL_DEPTH_8U, buff[0].c);

    int count = 0;
    demo_time = what_time_is_it_now();
    
   // CvVideoWriter *writer = NULL;
    FILE *fp = fopen(whitelist_name,"r");
    if(fp != NULL){
        char buffer[1024];
        while(fgets(buffer,sizeof(buffer),fp) != NULL){ 
            listsize++;
        }
        fclose(fp);
        fp = fopen(whitelist_name,"r");
        if(listsize != 0){
            int refined_listsize = 0;
            whitelist = calloc(listsize, sizeof(char*));
            for(int i = 0; i < listsize && fgets(buffer,1024,fp) != NULL; i++){
                if(strcmp(buffer,"\n") == 0){ 
                    continue;
                }
                else if(buffer[strlen(buffer) - 1] == '\n'){
                    buffer[strlen(buffer) - 1] = '\0';
                }
                whitelist[refined_listsize++] = strdup(buffer);
            }
            listsize = refined_listsize;
        }
        fclose(fp);
    } 

    printf("white_list size : %d\n",listsize);    
    for(int i = 0; i < listsize; i++) printf("%d_%s\n",i,whitelist[i]);
    
    // create FILE for json result output
    char *outputtext_name = calloc(strlen(prefix) + 10,sizeof(char));
    sprintf(outputtext_name,"%s%s",prefix,".txt");
    FILE *outputtext = fopen(outputtext_name, "w");

    // create video writer for output video
    CvVideoWriter *writer = NULL;
    if(outputvideo){
        char *outputvideo_name = calloc(strlen(prefix) + strlen(outputvideo) + 10, sizeof(char));
        sprintf(outputvideo_name,"%s_%s",prefix,outputvideo);
        double fps = cvGetCaptureProperty(cap, CV_CAP_PROP_FPS);
        if(fps <= 0 || fps >= 30) fps = 30;
        writer = cvCreateVideoWriter(outputvideo_name, CV_FOURCC('M', 'P', '4', '2'), fps, cvSize(buff[0].w,buff[0].h), 1);
        free(outputvideo_name);
    }

    bool image_save_flag = true;
    while(!demo_done){
        // buff |_|_|_|  save_jpg : (buff_index + 1)%3 | fetch_image : (buff_index)%3 | detect_image : (buff_index + 2) % 3 
        // 1st:  buff_index=1; |d|f|s| save_jpg-counter=0, detect_image_json-counter=1;
        // 2nd:  buff_index=2; |s|d|f| save_jpg-counter=1, detect_image_json-counter=2;
        // 3rd:  buff_index=3; |f|s|d| save_jpg-counter=2, detect_image_json-counter=3;
        buff_index = (buff_index + 1) %3;
        if(pthread_create(&fetch_thread, 0, fetch_in_thread, 0)) error("Thread creation failed");
        if(pthread_create(&detect_thread, 0, detect_in_thread, 0)) error("Thread creation failed");
        
        // save previous image result
        image p = buff[(buff_index + 1)%3];
        if(outputvideo){
            // in output video mode, always write image result to video.
            save_image_video(writer, p);
        }
        else{
            // in output image mode, check image_save_flag to determine save image or not
            if(image_save_flag == true){
                char name[256];
                sprintf(name, "%s_%08d", prefix, count);
                //printf("save image : %s\n",name);
                save_image(p, name);
            }
        }

        pthread_join(fetch_thread, 0);
        pthread_join(detect_thread, 0);
        ++count;

        // save current frame detection json result.
        char name[256];
        sprintf(name, "%s_%08d.jpg", prefix, count);
        //printf("save json : %s\n",name);
        write_json_file(outputtext,name,tag_results);
        
        // if image contain object, set flag = true.
        if(strlen(tag_results) != 0)
            image_save_flag = true;
        else
            image_save_flag = false;
        
        memset(tag_results, '\0', strlen(tag_results));

        if(count % 2 == 0){
            int progress = 0;
            if(total_videoframes > 0)
                progress = (count * 100.0 / total_videoframes);
            else 
                progress = 0;
            update_progress(progress);
            fflush(outputtext);
            fflush(stdout);
            fflush(stderr);
        }
    }

    update_progress(100);
    output_result(outputtext_name);

    if(writer) cvReleaseVideoWriter(&writer);
    if(whitelist) {
        for(int i = 0; i < listsize; i++){
            free(whitelist[i]);
        }
        free(whitelist);
    }
    if(outputtext_name) free(outputtext_name);
    if(outputtext) fclose(outputtext);
}

void demo_compare(char *cfg1, char *weight1, char *cfg2, char *weight2, float thresh, int cam_index, const char *filename, char **names, int classes, int delay, char *prefix, int avg_frames, float hier, int w, int h, int frames, int fullscreen)
{
    demo_frame = avg_frames;
    predictions = calloc(demo_frame, sizeof(float*));
    image **alphabet = load_alphabet();
    demo_names = names;
    demo_alphabet = alphabet;
    demo_classes = classes;
    demo_thresh = thresh;
    demo_hier = hier;
    printf("Demo\n");
    net = load_network(cfg1, weight1, 0);
    set_batch_network(net, 1);
    pthread_t detect_thread;
    pthread_t fetch_thread;

    srand(2222222);

    if(filename){
        printf("video file: %s\n", filename);
        cap = cvCaptureFromFile(filename);
    }else{
        cap = cvCaptureFromCAM(cam_index);

        if(w){
            cvSetCaptureProperty(cap, CV_CAP_PROP_FRAME_WIDTH, w);
        }
        if(h){
            cvSetCaptureProperty(cap, CV_CAP_PROP_FRAME_HEIGHT, h);
        }
        if(frames){
            cvSetCaptureProperty(cap, CV_CAP_PROP_FPS, frames);
        }
    }

    if(!cap) error("Couldn't connect to webcam.\n");

    layer l = net->layers[net->n-1];
    demo_detections = l.n*l.w*l.h;
    int j;

    avg = (float *) calloc(l.outputs, sizeof(float));
    for(j = 0; j < demo_frame; ++j) predictions[j] = (float *) calloc(l.outputs, sizeof(float));

    boxes = (box *)calloc(l.w*l.h*l.n, sizeof(box));
    probs = (float **)calloc(l.w*l.h*l.n, sizeof(float *));
    for(j = 0; j < l.w*l.h*l.n; ++j) probs[j] = (float *)calloc(l.classes+1, sizeof(float));

    buff[0] = get_image_from_stream(cap);
    buff[1] = copy_image(buff[0]);
    buff[2] = copy_image(buff[0]);
    buff_letter[0] = letterbox_image(buff[0], net->w, net->h);
    buff_letter[1] = letterbox_image(buff[0], net->w, net->h);
    buff_letter[2] = letterbox_image(buff[0], net->w, net->h);
    ipl = cvCreateImage(cvSize(buff[0].w,buff[0].h), IPL_DEPTH_8U, buff[0].c);

    int count = 0;
    if(!prefix){
        cvNamedWindow("Demo", CV_WINDOW_NORMAL); 
        if(fullscreen){
            cvSetWindowProperty("Demo", CV_WND_PROP_FULLSCREEN, CV_WINDOW_FULLSCREEN);
        } else {
            cvMoveWindow("Demo", 0, 0);
            cvResizeWindow("Demo", 1352, 1013);
        }
    }

    demo_time = what_time_is_it_now();

    while(!demo_done){
        buff_index = (buff_index + 1) %3;
        if(pthread_create(&fetch_thread, 0, fetch_in_thread, 0)) error("Thread creation failed");
        if(pthread_create(&detect_thread, 0, detect_in_thread, 0)) error("Thread creation failed");
        if(!prefix){
            fps = 1./(what_time_is_it_now() - demo_time);
            demo_time = what_time_is_it_now();
            display_in_thread(0);
        }else{
            char name[256];
            sprintf(name, "%s_%08d", prefix, count);
            save_image(buff[(buff_index + 1)%3], name);
        }
        pthread_join(fetch_thread, 0);
        pthread_join(detect_thread, 0);
        ++count;
    }
}
#else
void demo(char *cfgfile, char *weightfile, float thresh, int cam_index, const char *filename, char **names, int classes, int delay, char *prefix, int avg, float hier, int w, int h, int frames, int fullscreen, const char *outputvideo)
{
    fprintf(stderr, "Demo needs OpenCV for webcam images.\n");
}
#endif

