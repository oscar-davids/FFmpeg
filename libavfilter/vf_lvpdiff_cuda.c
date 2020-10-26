/*
 * @file
 * Caculate the diffmatrix between two input videos.
*/

#include "config.h"
#if HAVE_OPENCV2_CORE_CORE_C_H
#include <opencv2/core/core_c.h>
#include <opencv2/imgproc/imgproc_c.h>
#else
#include <opencv/cv.h>
#include <opencv/cxcore.h>
#endif
#include "libavutil/time.h"
#include "libavutil/avstring.h"
#include "libavutil/opt.h"
#include "libavutil/pixdesc.h"
#include "libswscale/swscale.h"

#include "avfilter.h"
#include "drawutils.h"
#include "formats.h"
#include "framesync.h"
#include "internal.h"
#include "psnr.h"
#include "video.h"

#define MAX_SEGMENT_TIME	6	//6 second segment
#define MAX_SAMPLE_NUM		18	//18 randomize index
#define CKNUM_PER_SEC		3	//check frame count per second
#define NORMAL_WIDTH		480	//normalize width
#define NORMAL_HEIGHT		270	//normalize height
#define MAX_FEATURE_NUM		5	//final score array

/*
typedef struct FramePairList {
	int		width;
	int		height;
	int		normalw;
	int		normalh;
	int		samplecount;
	int		featurecount;
	void	**listmain;
	void	**listref;
	double  *diffmatrix; //used in Opencv engine
	double	*finalscore;
} FramePairList;
*/

typedef struct LVPDiffContext{
    const AVClass *class;
    FFFrameSync fs;    

    FILE *stats_file;
    char *stats_file_str;
	int   stats_version;
	
	uint64_t nb_frames; //total sync number
	int is_rgb;

	int fps;
    int checknumpersec;
	int normalw;
	int normalh;
	struct AVFrame      *swscaleframe1;
	struct AVFrame      *swscaleframe2;

	int				randomIdx[MAX_SAMPLE_NUM];
	FramePairList	compInfo;
	
	struct SwsContext   *sws_rgb_scale1;
	struct SwsContext   *sws_rgb_scale2;

} LVPDiffContext;

#define OFFSET(x) offsetof(LVPDiffContext, x)
#define FLAGS AV_OPT_FLAG_FILTERING_PARAM|AV_OPT_FLAG_VIDEO_PARAM

static const AVOption lvpdiff_cuda_options[] = {
    {"stats_file", "Set file where to store per-frame difference information", OFFSET(stats_file_str), AV_OPT_TYPE_STRING, {.str=NULL}, 0, 0, FLAGS },
    {"f",          "Set file where to store per-frame difference information", OFFSET(stats_file_str), AV_OPT_TYPE_STRING, {.str=NULL}, 0, 0, FLAGS },
    {"stats_version", "Set the format version for the stats file.",            OFFSET(stats_version),  AV_OPT_TYPE_INT,    {.i64=1},    1, 1, FLAGS },
    { NULL }
};

FRAMESYNC_DEFINE_CLASS(lvpdiff_cuda, LVPDiffContext, fs);

static void get_random(int* rdx, int fps,int count)
{
	int random, i, j;
	for (int i = 0; i < count; i++){	
		//random = 0;
		//while(!random)
		//	random = rand() % fps;
		rdx[i] = rand() % fps;
	}
	//sort indexs
	for (i = 0; i < count - 1; i++) {
		for (j = i + 1; j < count; j++) {
			if (rdx[i] > rdx[j]) {
				random = rdx[i];
				rdx[i] = rdx[j];
				rdx[j] = random;
			}
		}
	}
}

static void init_randomidx(LVPDiffContext *s)
{
	time_t t;
	int rdx[CKNUM_PER_SEC] = { 0, };
	//srand((unsigned)time(&t));
	srand(av_gettime());
	for (int i = 0; i < MAX_SEGMENT_TIME; i++){	
		get_random(rdx, s->fps, CKNUM_PER_SEC);
		for (int j = 0; j < CKNUM_PER_SEC; j++){
			
			s->randomIdx[i*CKNUM_PER_SEC + j] = s->fps * i + rdx[j];
			//debug oscar
			s->randomIdx[i*CKNUM_PER_SEC + j] = s->fps * i + j * (s->fps / 4);
			//debug oscar
			//av_log(NULL, AV_LOG_DEBUG, "master rand idx(%d) = %d\n", i*CKNUM_PER_SEC + j, s->randomIdx[i*CKNUM_PER_SEC + j]);
		}
	}	

}

int avfilter_run_calcdiffmatrix_cuda(void *framebufflist)
{
	int ret = 0;	
	ret	= cvCalcDiffMatrixwithCuda(framebufflist);
	return ret;
}

static int query_formats(AVFilterContext *ctx)
{
    static const enum AVPixelFormat pix_fmts[] = {
        AV_PIX_FMT_GRAY8, AV_PIX_FMT_GRAY9, AV_PIX_FMT_GRAY10, AV_PIX_FMT_GRAY12, AV_PIX_FMT_GRAY14, AV_PIX_FMT_GRAY16,
#define PF_NOALPHA(suf) AV_PIX_FMT_YUV420##suf,  AV_PIX_FMT_YUV422##suf,  AV_PIX_FMT_YUV444##suf
#define PF_ALPHA(suf)   AV_PIX_FMT_YUVA420##suf, AV_PIX_FMT_YUVA422##suf, AV_PIX_FMT_YUVA444##suf
#define PF(suf)         PF_NOALPHA(suf), PF_ALPHA(suf)
        PF(P), PF(P9), PF(P10), PF_NOALPHA(P12), PF_NOALPHA(P14), PF(P16),
        AV_PIX_FMT_YUV440P, AV_PIX_FMT_YUV411P, AV_PIX_FMT_YUV410P,
        AV_PIX_FMT_YUVJ411P, AV_PIX_FMT_YUVJ420P, AV_PIX_FMT_YUVJ422P,
        AV_PIX_FMT_YUVJ440P, AV_PIX_FMT_YUVJ444P,
        AV_PIX_FMT_GBRP, AV_PIX_FMT_GBRP9, AV_PIX_FMT_GBRP10,
        AV_PIX_FMT_GBRP12, AV_PIX_FMT_GBRP14, AV_PIX_FMT_GBRP16,
        AV_PIX_FMT_GBRAP, AV_PIX_FMT_GBRAP10, AV_PIX_FMT_GBRAP12, AV_PIX_FMT_GBRAP16,
        AV_PIX_FMT_NONE
    };

    AVFilterFormats *fmts_list = ff_make_format_list(pix_fmts);
    if (!fmts_list)
        return AVERROR(ENOMEM);
    return ff_set_common_formats(ctx, fmts_list);
}

static int is_checkframe(LVPDiffContext *s, int idx){
	int ret =0;
	for (int i = 0; i < MAX_SAMPLE_NUM; i++){ 
		if(idx == s->randomIdx[i]){
			ret = 1;
			break;
		}
	}
	return ret;
}

static int do_lvpdiff(FFFrameSync *fs)
{
    AVFilterContext *ctx = fs->parent;
    LVPDiffContext *s = ctx->priv;
    AVFrame *master, *ref;    
    AVDictionary **metadata;
	int ret;
	char value[64] = {0,};
    char dickey[64] = "lavfi.lvpdiff";
	

	//debug oscar
	//av_log(NULL, AV_LOG_ERROR, "do_lvpdiff %d\n",ctx->outputs[0]->frame_count_in);

    ret = ff_framesync_dualinput_get(fs, &master, &ref);
    if (ret < 0)
        return ret;
	
    if (!ref)
        return ff_filter_frame(ctx->outputs[0], master);

	if(is_checkframe(s,ctx->outputs[0]->frame_count_in)){
		//make compair frames
		sws_scale(s->sws_rgb_scale1, (const uint8_t **)master->data, master->linesize,
                  0, master->height, (uint8_t * const*)(&s->swscaleframe1->data),
                  s->swscaleframe1->linesize);
		sws_scale(s->sws_rgb_scale2, (const uint8_t **)ref->data, ref->linesize,
                  0, ref->height, (uint8_t * const*)(&s->swscaleframe2->data),
                  s->swscaleframe2->linesize);

		void* ptmp = (void*)malloc(s->normalw*s->normalh * 3);
		memcpy(ptmp, s->swscaleframe1->data[0], s->normalw*s->normalh * 3);
		s->compInfo.listmain[s->compInfo.samplecount] = ptmp;
		
		ptmp = (void*)malloc(s->normalw*s->normalh * 3);
		memcpy(ptmp, s->swscaleframe2->data[0], s->normalw*s->normalh * 3);
		s->compInfo.listref[s->compInfo.samplecount] = ptmp;
		
		s->compInfo.samplecount++;
		//av_log(NULL, AV_LOG_DEBUG, "do_lvpdiff grab comapair %d\n", s->compInfo.samplecount);
	}
	
	s->nb_frames++;
	
	metadata = &master->metadata;
	//make metadata for lvpdiff
	sprintf(value,"%04d",s->nb_frames);	
	av_dict_set(metadata, (const char*)dickey, (const char*)value, 0);    

    return ff_filter_frame(ctx->outputs[0], master);
}
static av_cold int init(AVFilterContext *ctx)
{
	int ret;
    LVPDiffContext *s = ctx->priv;	

	s->normalw = NORMAL_WIDTH;
	s->normalh = NORMAL_HEIGHT;

	//master scale frame
	s->swscaleframe1 = av_frame_alloc();
    if (!s->swscaleframe1)
        return AVERROR(ENOMEM);

    s->swscaleframe1->format = AV_PIX_FMT_BGR24;
    s->swscaleframe1->width  = s->normalw;
    s->swscaleframe1->height = s->normalh;
    ret = av_frame_get_buffer(s->swscaleframe1, 0);
    if (ret < 0) {
        av_frame_free(&s->swscaleframe1);
        return ret;
    }
	//slave scale frame
	s->swscaleframe2 = av_frame_alloc();
    if (!s->swscaleframe2)
        return AVERROR(ENOMEM);

    s->swscaleframe2->format = AV_PIX_FMT_BGR24;
    s->swscaleframe2->width  = s->normalw;
    s->swscaleframe2->height = s->normalh;
    ret = av_frame_get_buffer(s->swscaleframe2, 0);
    if (ret < 0) {
        av_frame_free(&s->swscaleframe2);
        return ret;
    }

	s->compInfo.samplecount = 0;
	s->compInfo.listmain = (void**)malloc(sizeof(void*)*MAX_SAMPLE_NUM);
	s->compInfo.listref = (void**)malloc(sizeof(void*)*MAX_SAMPLE_NUM);

    if (s->stats_file_str) {

        if (!strcmp(s->stats_file_str, "-")) {
            s->stats_file = stdout;
        } else {
            s->stats_file = fopen(s->stats_file_str, "w");
            if (!s->stats_file) {
                int err = AVERROR(errno);
                char buf[128];
                av_strerror(err, buf, sizeof(buf));
                av_log(ctx, AV_LOG_ERROR, "Could not open stats file %s: %s\n",
                       s->stats_file_str, buf);
                return err;
            }
        }
    }

    s->fs.on_event = do_lvpdiff;
    return 0;
}

static int config_input_ref(AVFilterLink *inlink)
{
    const AVPixFmtDescriptor *desc = av_pix_fmt_desc_get(inlink->format);
    AVFilterContext *ctx  = inlink->dst;
    LVPDiffContext *s = ctx->priv;
    double average_max;
    unsigned sum;
    int j;
	
	AVFilterLink *mainlink = ctx->inputs[0];

	//get fps
	s->fps = 0;
	float vinfofps = 1.0;
	if (mainlink->frame_rate.den > 0.0) {
		s->fps = (int)(av_q2d(mainlink->frame_rate) + 0.5);
	}
	else {
		s->fps = (int)(1.0 / av_q2d(mainlink->time_base) + 0.5);
	}

	s->sws_rgb_scale1 = sws_getContext(ctx->inputs[0]->w, ctx->inputs[0]->h, ctx->inputs[0]->format,
		s->normalw, s->normalh, AV_PIX_FMT_BGR24,
		SWS_BILINEAR, NULL, NULL, NULL);

	s->sws_rgb_scale2 = sws_getContext(ctx->inputs[1]->w, ctx->inputs[1]->h, ctx->inputs[1]->format,
		s->normalw, s->normalh, AV_PIX_FMT_BGR24,
		SWS_BILINEAR, NULL, NULL, NULL);

	init_randomidx(s);	

	s->compInfo.width = ctx->inputs[1]->w;
	s->compInfo.height = ctx->inputs[1]->h;

	s->compInfo.normalw = NORMAL_WIDTH;
	s->compInfo.normalh = NORMAL_HEIGHT;

	s->compInfo.featurecount = MAX_FEATURE_NUM;

	//debug oscar
	av_log(NULL, AV_LOG_DEBUG, "master fps = %d w =%d h=%d \n", s->fps, s->compInfo.width, s->compInfo.height);

    return 0;
}

static int config_output(AVFilterLink *outlink)
{
    AVFilterContext *ctx = outlink->src;
    LVPDiffContext *s = ctx->priv;
    AVFilterLink *mainlink = ctx->inputs[0];
    int ret;

    ret = ff_framesync_init_dualinput(&s->fs, ctx);
    if (ret < 0)
        return ret;
    outlink->w = mainlink->w;
    outlink->h = mainlink->h;
    outlink->time_base = mainlink->time_base;
    outlink->sample_aspect_ratio = mainlink->sample_aspect_ratio;
    outlink->frame_rate = mainlink->frame_rate;
    if ((ret = ff_framesync_configure(&s->fs)) < 0)
        return ret;

    outlink->time_base = s->fs.time_base;

    if (av_cmp_q(mainlink->time_base, outlink->time_base) &&
        av_cmp_q(ctx->inputs[1]->time_base, outlink->time_base))
        av_log(ctx, AV_LOG_WARNING, "not matching timebases found between first input: %d/%d and second input %d/%d, results may be incorrect!\n",
               mainlink->time_base.num, mainlink->time_base.den,
               ctx->inputs[1]->time_base.num, ctx->inputs[1]->time_base.den);

    return 0;
}

static int activate(AVFilterContext *ctx)
{
    LVPDiffContext *s = ctx->priv;
    return ff_framesync_activate(&s->fs);
}

static av_cold void uninit(AVFilterContext *ctx)
{
	int i;
    LVPDiffContext *s = ctx->priv;
	
	if(s->compInfo.samplecount > 0){
		//call opencv api at here
		
		//creat feature matrix(feature * samplecount)	
		s->compInfo.diffmatrix = (double*)malloc(sizeof(double) * s->compInfo.featurecount * s->compInfo.samplecount);
		memset(s->compInfo.diffmatrix, 0x00, sizeof(double) * s->compInfo.featurecount * s->compInfo.samplecount);
		//creat final score buffer
		s->compInfo.finalscore = (double*)malloc(sizeof(double)*s->compInfo.featurecount);

		cvCalcDiffMatrixwithCuda((void*)&s->compInfo);

		//debug oscar
		av_log(NULL, AV_LOG_ERROR, "do_lvpdiff_cuda comapare frame count %d\n", s->compInfo.samplecount);
		for(i = 0 ; i < s->compInfo.featurecount; i++){
			av_log(NULL, AV_LOG_ERROR, "feature(%d) = %lf\n", i, s->compInfo.finalscore[i]);
		}

		if (s->stats_file) {
			for (i = 0; i < s->compInfo.featurecount; i++) {				
				fprintf(s->stats_file, "feature(%02d):%0.lf ",i , s->compInfo.finalscore[i]);
			}			
			fprintf(s->stats_file, "\n");
		}

		for(i = 0 ; i < s->compInfo.samplecount ; i++){
			if(s->compInfo.listmain[i]) free(s->compInfo.listmain[i]);
			if(s->compInfo.listref[i]) free(s->compInfo.listref[i]);
		}
		if(s->compInfo.listmain) free(s->compInfo.listmain);
		if(s->compInfo.listref) free(s->compInfo.listref);
		if (s->compInfo.diffmatrix) free(s->compInfo.diffmatrix);
		if (s->compInfo.finalscore) free(s->compInfo.finalscore);
	}

	//free master & slave scale frames
	if(s->swscaleframe1)
        av_frame_free(&s->swscaleframe1);
	if(s->swscaleframe2)
        av_frame_free(&s->swscaleframe2);

    ff_framesync_uninit(&s->fs);

    if (s->stats_file && s->stats_file != stdout)
        fclose(s->stats_file);
}

static const AVFilterPad lvpdiff_inputs[] = {
    {
        .name         = "main",
        .type         = AVMEDIA_TYPE_VIDEO,
    },{
        .name         = "reference",
        .type         = AVMEDIA_TYPE_VIDEO,
        .config_props = config_input_ref,
    },
    { NULL }
};

static const AVFilterPad lvpdiff_outputs[] = {
    {
        .name          = "default",
        .type          = AVMEDIA_TYPE_VIDEO,
        .config_props  = config_output,
    },
    { NULL }
};

AVFilter ff_vf_lvpdiff_cuda = {
    .name          = "lvpdiff_cuda",
    .description   = NULL_IF_CONFIG_SMALL("Calculate the lvpdiff between two video streams."),
    .preinit       = lvpdiff_cuda_framesync_preinit,
    .init          = init,
    .uninit        = uninit,
    .query_formats = query_formats,
    .activate      = activate,
    .priv_size     = sizeof(LVPDiffContext),
    .priv_class    = &lvpdiff_cuda_class,
    .inputs        = lvpdiff_inputs,
    .outputs       = lvpdiff_outputs,
};
