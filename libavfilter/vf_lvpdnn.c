/*
 * Copyright (c) 2020 oscar
 *
 * This file is part of FFmpeg.
 *
 * FFmpeg is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * FFmpeg is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with FFmpeg; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA
 */

/**
 * @file 
 * implementing livepeer frame filter using deep convolutional networks. 
 */

#include "libavformat/avio.h"
#include "libavutil/opt.h"
#include "libavutil/pixdesc.h"
#include "libavutil/avassert.h"
#include "libavutil/imgutils.h"
#include "libavutil/hwcontext.h"
#include "libavutil/hwcontext_internal.h"
#include "libswscale/swscale.h"
#include "avfilter.h"
#include "dnn_interface.h"
#include "formats.h"
#include "internal.h"


typedef struct LVPDnnContext {
    const AVClass *class;

    int     filter_type;
    char    *model_filename;
    DNNBackendType backend_type;
    char    *model_inputname;
    char    *model_outputname;
    int     sample_rate;
    double  valid_threshold;
    char    *log_filename;

    DNNModule   *dnn_module;
    DNNModel    *model;

    // input & output of the model at execution time
    DNNData input;
    DNNData output;
    
    struct SwsContext   *sws_rgb_scale;
    struct SwsContext   *sws_gray8_to_grayf32;

    struct AVFrame      *swscaleframe;
    struct AVFrame      *swframeforHW;

    FILE                *logfile;
    int                 framenum;
    //
    struct SwsContext   *sws_grayf32_to_gray8;    
    int                 sws_uv_height;

} LVPDnnContext;

#define OFFSET(x) offsetof(LVPDnnContext, x)
#define FLAGS AV_OPT_FLAG_FILTERING_PARAM | AV_OPT_FLAG_VIDEO_PARAM
static const AVOption lvpdnn_options[] = {
    { "filter_type", "filter type(inference/detect)",   OFFSET(filter_type),    AV_OPT_TYPE_INT,        { .i64 = 0 },    0, 1, FLAGS, "type" },
    { "inference",   "inference filter flag",           0,                      AV_OPT_TYPE_CONST,      { .i64 = 0 },    0, 0, FLAGS, "type" },
    { "detect",      "detect filter flag",              0,                      AV_OPT_TYPE_CONST,      { .i64 = 1 },    0, 0, FLAGS, "type" },
    { "dnn_backend", "DNN backend",                OFFSET(backend_type),     AV_OPT_TYPE_INT,           { .i64 = 0 },    0, 1, FLAGS, "backend" },
    { "native",      "native backend flag",        0,                        AV_OPT_TYPE_CONST,         { .i64 = 0 },    0, 0, FLAGS, "backend" },
#if (CONFIG_LIBTENSORFLOW == 1)
    { "tensorflow",  "tensorflow backend flag",    0,                        AV_OPT_TYPE_CONST,         { .i64 = 1 },    0, 0, FLAGS, "backend" },
#endif
    { "model",       "path to model file",          OFFSET(model_filename),   AV_OPT_TYPE_STRING,       { .str = NULL }, 0, 0, FLAGS },
    { "input",       "input name of the model",     OFFSET(model_inputname),  AV_OPT_TYPE_STRING,       { .str = NULL }, 0, 0, FLAGS },
    { "output",      "output name of the model",    OFFSET(model_outputname), AV_OPT_TYPE_STRING,       { .str = NULL }, 0, 0, FLAGS },
    { "sample","detector one every sample frames",  OFFSET(sample_rate),    AV_OPT_TYPE_INT,            { .i64 = 1   },  0, 2000, FLAGS },
    { "threshold",  "threshold for verify",         OFFSET(valid_threshold),  AV_OPT_TYPE_DOUBLE,       { .dbl = 0.5 },  0, 2, FLAGS },
    { "log",         "path name of the log",        OFFSET(log_filename), AV_OPT_TYPE_STRING,           { .str = NULL }, 0, 0, FLAGS },    
    { NULL }
};

AVFILTER_DEFINE_CLASS(lvpdnn);

static av_cold int init(AVFilterContext *context)
{
    LVPDnnContext *ctx = context->priv;

    if (!ctx->model_filename) {
        av_log(ctx, AV_LOG_ERROR, "model file for network is not specified\n");
        return AVERROR(EINVAL);
    }
    if (!ctx->model_inputname) {
        av_log(ctx, AV_LOG_ERROR, "input name of the model network is not specified\n");
        return AVERROR(EINVAL);
    }
    if (!ctx->model_outputname) {
        av_log(ctx, AV_LOG_ERROR, "output name of the model network is not specified\n");
        return AVERROR(EINVAL);
    }

    if (!ctx->log_filename) {
        av_log(ctx, AV_LOG_INFO, "output file for log is not specified\n");
        //return AVERROR(EINVAL);
    }

    ctx->backend_type = 1;
    ctx->dnn_module = ff_get_dnn_module(ctx->backend_type);
    if (!ctx->dnn_module) {
        av_log(ctx, AV_LOG_ERROR, "could not create DNN module for requested backend\n");
        return AVERROR(ENOMEM);
    }
    if (!ctx->dnn_module->load_model) {
        av_log(ctx, AV_LOG_ERROR, "load_model for network is not specified\n");
        return AVERROR(EINVAL);
    }

    ctx->model = (ctx->dnn_module->load_model)(ctx->model_filename);
    if (!ctx->model) {
        av_log(ctx, AV_LOG_ERROR, "could not load DNN model\n");
        return AVERROR(EINVAL);
    }

    if(ctx->log_filename){        
        ctx->logfile = fopen(ctx->log_filename, "w");
    }
    else{        
        ctx->logfile = NULL;
    }

    ctx->framenum = 0;

    return 0;
}

static int query_formats(AVFilterContext *context)
{
    static const enum AVPixelFormat pix_fmts[] = {
        AV_PIX_FMT_RGB24, AV_PIX_FMT_BGR24,AV_PIX_FMT_GRAY8, AV_PIX_FMT_GRAYF32,
        AV_PIX_FMT_YUV420P, AV_PIX_FMT_YUV422P, AV_PIX_FMT_YUV444P,
        AV_PIX_FMT_YUV410P, AV_PIX_FMT_YUV411P,
        AV_PIX_FMT_CUDA,AV_PIX_FMT_NONE
    };
    AVFilterFormats *fmts_list = ff_make_format_list(pix_fmts);
    return ff_set_common_formats(context, fmts_list);
}

static int prepare_sws_context(AVFilterLink *inlink)
{
    int result = 0;
    enum AVPixelFormat fmt = inlink->format;
    AVFilterContext *context    = inlink->dst;
    LVPDnnContext *ctx = context->priv;    
    DNNDataType input_dt  = ctx->input.dt; 

    //check hwframe 
    if (inlink->hw_frames_ctx)
    {
        enum AVPixelFormat *formats;

        result = av_hwframe_transfer_get_formats(inlink->hw_frames_ctx,
                                            AV_HWFRAME_TRANSFER_DIRECTION_FROM,
                                            &formats, 0);
        if(result <0 ){
            av_log(ctx, AV_LOG_ERROR, "could not find HW Pixelformat for scale\n");
            return result;
        }

        fmt = formats[0];            
        av_freep(&formats);
    }

    av_assert0(input_dt == DNN_FLOAT);     

    ctx->sws_rgb_scale = sws_getContext(inlink->w, inlink->h, fmt,
                                            ctx->input.width, ctx->input.height, AV_PIX_FMT_RGB24,
                                            SWS_BILINEAR, NULL, NULL, NULL);

    ctx->sws_gray8_to_grayf32 = sws_getContext(ctx->input.width*3,
                                                ctx->input.height,
                                                AV_PIX_FMT_GRAY8,
                                                ctx->input.width*3,
                                                ctx->input.height,
                                                AV_PIX_FMT_GRAYF32,
                                                0, NULL, NULL, NULL);  

  

    if(ctx->sws_rgb_scale == 0 || ctx->sws_gray8_to_grayf32 == 0)
    {
        av_log(ctx, AV_LOG_ERROR, "could not create scale context\n");
        return AVERROR(ENOMEM);
    }

    ctx->swscaleframe = av_frame_alloc();
    if (!ctx->swscaleframe)
        return AVERROR(ENOMEM);

    ctx->swscaleframe->format = AV_PIX_FMT_RGB24;
    ctx->swscaleframe->width  = ctx->input.width;
    ctx->swscaleframe->height = ctx->input.height;
    result = av_frame_get_buffer(ctx->swscaleframe, 0);
    if (result < 0) {
        av_frame_free(&ctx->swscaleframe);
        return result;
    }

    if (inlink->hw_frames_ctx){
        ctx->swframeforHW = av_frame_alloc();

        if (!ctx->swframeforHW)
        return AVERROR(ENOMEM);
    }

    return 0;
}
static int config_input(AVFilterLink *inlink)
{
    AVFilterContext *context     = inlink->dst;
    LVPDnnContext *ctx = context->priv;
    DNNReturnType result;
    DNNData model_input;
    int check;    

    result = ctx->model->get_input(ctx->model->model, &model_input, ctx->model_inputname);
    if (result != DNN_SUCCESS) {
        av_log(ctx, AV_LOG_ERROR, "could not get input from the model\n");
        return AVERROR(EIO);
    }

    ctx->input.width    = model_input.width;
    ctx->input.height   = model_input.height;
    ctx->input.channels = model_input.channels;
    ctx->input.dt = model_input.dt;
    
    check = prepare_sws_context(inlink);
    if (check != 0) {
        av_log(ctx, AV_LOG_ERROR, "could not create scale context for the model\n");
        return AVERROR(EIO);
    }

    result = (ctx->model->set_input_output)(ctx->model->model,
                                        &ctx->input, ctx->model_inputname,
                                        (const char **)&ctx->model_outputname, 1);
    

    if (result != DNN_SUCCESS) {
        av_log(ctx, AV_LOG_ERROR, "could not set input and output for the model\n");
        return AVERROR(EIO);
    }

    return 0;
}


static int config_output(AVFilterLink *outlink)
{
    AVFilterContext *context = outlink->src;
    LVPDnnContext *ctx = context->priv;
    DNNReturnType result;

    // have a try run in case that the dnn model resize the frame
    result = (ctx->dnn_module->execute_model)(ctx->model, &ctx->output, 1);
    if (result != DNN_SUCCESS){
        av_log(ctx, AV_LOG_ERROR, "failed to execute model\n");
        return AVERROR(EIO);
    }

    return 0;
}

static int copy_from_frame_to_dnn(LVPDnnContext *ctx, const AVFrame *frame)
{
    av_assert0(ctx->swscaleframe != 0);
    int bytewidth = av_image_get_linesize(ctx->swscaleframe->format, ctx->swscaleframe->width, 0);
    DNNData *dnn_input = &ctx->input;

    if(ctx->swframeforHW)
    {
        if(av_hwframe_transfer_data(ctx->swframeforHW, frame, 0) != 0)
            return AVERROR(EIO);
        
        sws_scale(ctx->sws_rgb_scale, (const uint8_t **)ctx->swframeforHW->data, ctx->swframeforHW->linesize,
                  0, ctx->swframeforHW->height, (uint8_t * const*)(&ctx->swscaleframe->data),
                 ctx->swscaleframe->linesize);

    }
    else
    {
        sws_scale(ctx->sws_rgb_scale, (const uint8_t **)frame->data, frame->linesize,
                  0, frame->height, (uint8_t * const*)(&ctx->swscaleframe->data),
                  ctx->swscaleframe->linesize);
    }


    if (dnn_input->dt == DNN_FLOAT) {
        sws_scale(ctx->sws_gray8_to_grayf32, (const uint8_t **)ctx->swscaleframe->data, ctx->swscaleframe->linesize,
                    0, ctx->swscaleframe->height, (uint8_t * const*)(&dnn_input->data),
                    (const int [4]){ctx->swscaleframe->width * 3 * sizeof(float), 0, 0, 0});
    } else {
        av_assert0(dnn_input->dt == DNN_UINT8);
        av_image_copy_plane(dnn_input->data, bytewidth,
                            ctx->swscaleframe->data[0], ctx->swscaleframe->linesize[0],
                            bytewidth, ctx->swscaleframe->height);
    }
    
    return 0;   
}

static int copy_from_dnn_to_frame(LVPDnnContext *ctx, AVFrame *frame)
{
    int bytewidth = av_image_get_linesize(frame->format, frame->width, 0);
    DNNData *dnn_output = &ctx->output;

    switch (frame->format) {
    case AV_PIX_FMT_RGB24:
    case AV_PIX_FMT_BGR24:
        if (dnn_output->dt == DNN_FLOAT) {
            sws_scale(ctx->sws_grayf32_to_gray8, (const uint8_t *[4]){(const uint8_t *)dnn_output->data, 0, 0, 0},
                      (const int[4]){frame->width * 3 * sizeof(float), 0, 0, 0},
                      0, frame->height, (uint8_t * const*)frame->data, frame->linesize);

        } else {
            av_assert0(dnn_output->dt == DNN_UINT8);
            av_image_copy_plane(frame->data[0], frame->linesize[0],
                                dnn_output->data, bytewidth,
                                bytewidth, frame->height);
        }
        return 0;
    case AV_PIX_FMT_GRAY8:
        // it is possible that data type of dnn output is float32,
        // need to add support for such case when needed.
        av_assert0(dnn_output->dt == DNN_UINT8);
        av_image_copy_plane(frame->data[0], frame->linesize[0],
                            dnn_output->data, bytewidth,
                            bytewidth, frame->height);
        return 0;
    case AV_PIX_FMT_GRAYF32:
        av_assert0(dnn_output->dt == DNN_FLOAT);
        av_image_copy_plane(frame->data[0], frame->linesize[0],
                            dnn_output->data, bytewidth,
                            bytewidth, frame->height);
        return 0;
    case AV_PIX_FMT_YUV420P:
    case AV_PIX_FMT_YUV422P:
    case AV_PIX_FMT_YUV444P:
    case AV_PIX_FMT_YUV410P:
    case AV_PIX_FMT_YUV411P:
        sws_scale(ctx->sws_grayf32_to_gray8, (const uint8_t *[4]){(const uint8_t *)dnn_output->data, 0, 0, 0},
                  (const int[4]){frame->width * sizeof(float), 0, 0, 0},
                  0, frame->height, (uint8_t * const*)frame->data, frame->linesize);
        return 0;
    default:
        return AVERROR(EIO);
    }

    return 0;
}

static int filter_frame(AVFilterLink *inlink, AVFrame *in)
{
    char slvpinfo[256] = {0,};
    AVFilterContext *context  = inlink->dst;
    AVFilterLink *outlink = context->outputs[0];
    LVPDnnContext *ctx = context->priv;
    DNNReturnType dnn_result;
    AVDictionary **metadata = &in->metadata;

    ctx->framenum ++;

    if(ctx->sample_rate > 0 && ctx->framenum % ctx->sample_rate == 0)
    {
        copy_from_frame_to_dnn(ctx, in);

        dnn_result = (ctx->dnn_module->execute_model)(ctx->model, &ctx->output, 1);
        if (dnn_result != DNN_SUCCESS){
            av_log(ctx, AV_LOG_ERROR, "failed to execute model\n");
            av_frame_free(&in);
            return AVERROR(EIO);
        }

        DNNData *dnn_output = &ctx->output;
        float* pfdata = dnn_output->data;
        int lendata = ctx->output.height;
        //            
        if(lendata >= 2 && pfdata[0] >= ctx->valid_threshold)
        {
            snprintf(slvpinfo, sizeof(slvpinfo), "probability %.2f", pfdata[0]);  
           
            av_dict_set(metadata, "lavfi.lvpdnn.text", slvpinfo, 0);
            if(ctx->logfile)
            {
                fprintf(ctx->logfile,"%s\n",slvpinfo);                
            }      
        }
        //for DEBUG
        //av_log(0, AV_LOG_INFO, "frame contents seems like aaa as %s\n",slvpinfo);        
    }

    if(ctx->logfile && ctx->framenum % 20 == 0)
        fflush(ctx->logfile);

    return ff_filter_frame(outlink, in);   
}

static av_cold void uninit(AVFilterContext *ctx)
{
    LVPDnnContext *context = ctx->priv;

    sws_freeContext(context->sws_rgb_scale);
    sws_freeContext(context->sws_gray8_to_grayf32);
    sws_freeContext(context->sws_grayf32_to_gray8);

    if (context->dnn_module)
        (context->dnn_module->free_model)(&context->model);

    av_freep(&context->dnn_module);

    if(context->swscaleframe)
        av_frame_free(&context->swscaleframe);

    if(context->swframeforHW)
        av_frame_free(&context->swframeforHW);

    if(context->log_filename && context->logfile)
    {
        fclose(context->logfile);        
    }
}

static const AVFilterPad lvpdnn_inputs[] = {
    {
        .name         = "default",
        .type         = AVMEDIA_TYPE_VIDEO,
        .config_props = config_input,
        .filter_frame = filter_frame,
    },
    { NULL }
};

static const AVFilterPad lvpdnn_outputs[] = {
    {
        .name = "default",
        .type = AVMEDIA_TYPE_VIDEO,
        .config_props  = config_output,
    },
    { NULL }
};

AVFilter ff_vf_lvpdnn = {
    .name          = "lvpdnn",
    .description   = NULL_IF_CONFIG_SMALL("Apply lvpdnn filter to the input."),
    .priv_size     = sizeof(LVPDnnContext),
    .init          = init,
    .uninit        = uninit,
    .query_formats = query_formats,
    .inputs        = lvpdnn_inputs,
    .outputs       = lvpdnn_outputs,
    .priv_class    = &lvpdnn_class,
};
