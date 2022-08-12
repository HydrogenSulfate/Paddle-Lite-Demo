package com.baidu.paddle.lite.demo.pp_shitu;

import android.graphics.Bitmap;
import android.util.Log;

public class Native {
    static {
        System.loadLibrary("Native");
    }

    protected Bitmap inputImage = null;
    protected long[] detinputShape = new long[]{1, 3, 640, 640};
    protected long[] recinputShape = new long[]{1, 3, 224, 224};
    protected float inferenceTime = 0;
    protected String top1Result = "";
    protected String top2Result = "";
    protected String top3Result = "";
    protected int topk = 1;
    protected boolean addGallery = false;
    protected String label_name = "";
    protected boolean clearFeature = false;
    private long ctx = 0;

    public boolean init(String DetModelDir,
                        String RecModelDir,
                        String labelPath,
                        String IndexDir,
                        long[] DetInputShape,
                        long[] RecInputShape,
                        int cpuThreadNum,
                        int WarmUp,
                        int Repeat,
                        int topk,
                        boolean add_gallery,
//                        boolean clear_feature,
                        String cpuMode) {
        if (DetInputShape.length != 4) {
            Log.i("Paddle-lite", "Size of input shape should be: 4");
            return false;
        }
        if (DetInputShape[0] != 1) {
            Log.i("Paddle-lite", "Only one batch is supported in the image classification demo, you can use any batch size in " +
                    "your Apps!");
            return false;
        }
        if (DetInputShape[1] != 1 && DetInputShape[1] != 3) {
            Log.i("Paddle-lite", "Only one/three channels are supported in the image classification demo, you can use any " +
                    "channel size in your Apps!");
            return false;
        }
        this.detinputShape = DetInputShape;
        this.recinputShape = RecInputShape;
        this.topk = topk;
        this.addGallery = add_gallery;
//        this.clearFeature = clear_feature;
        ctx = nativeInit(
                DetModelDir,
                RecModelDir,
                labelPath,
                IndexDir,
                DetInputShape,
                RecInputShape,
                cpuThreadNum,
                WarmUp,
                Repeat,
                topk,
                add_gallery,
//                clear_feature,
                cpuMode);
        return ctx != 0;
    }

    public boolean release() {
        if (ctx == 0) {
            return false;
        }
        return nativeRelease(ctx);
    }

    public void setAddGallery(boolean flag) {
        nativesetAddGallery(ctx, flag);
    }

    public void saveIndex(String save_file_name) {
        nativesaveIndex(ctx, save_file_name);
    }

    public boolean loadIndex(String load_file_name) {
        return nativeloadIndex(ctx, load_file_name);
    }

    public void clearFeature() {
        boolean ret = nativeclearGallery(ctx);
    }

    public void setLabelName(String label_name) {
        this.label_name = label_name;
    }

    public boolean process() {
        if (ctx == 0) {
            return false;
        }
        // ARGB8888 bitmap is only supported in native, other color formats can be added by yourself.
        String[] res = nativeProcess(ctx, this.inputImage, label_name).split("\n");
        if (res.length >= 1) {
            inferenceTime = Float.parseFloat(res[0]);
            if (res.length >= 2){
                top1Result = res[1];
            } else {
                top1Result = "";
            }
            if (res.length >= 3 && res.length <= this.topk + 1) {
                top2Result = res[2];
            } else {
                top2Result = "";
            }
            if (res.length >= 4 && res.length <= this.topk + 1) {
                top3Result = res[3];
            } else {
                top3Result = "";
            }
        }
        return (res.length > 0);
    }
    public boolean isLoaded() {
        return ctx != 0;
    }
    public void setInputImage(Bitmap image) {
        if (image == null) {
            return;
        }
        Bitmap rgbaImage = image.copy(Bitmap.Config.ARGB_8888, true);
        this.inputImage = rgbaImage;
    }
    public float inferenceTime() {
        return inferenceTime;
    }

    public Bitmap inputImage() {
        return inputImage;
    }

    public String top1Result() {
        return top1Result;
    }

    public String top2Result() {
        return top2Result;
    }

    public String top3Result() {
        return top3Result;
    }

    public static native long nativeInit(String DetModelDir,
                                         String RecModelDir,
                                         String labelPath,
                                         String IndexDir,
                                         long[] DetInputShape,
                                         long[] RecInputShape,
                                         int cpuThreadNum,
                                         int WarmUp,
                                         int Repeat,
                                         int topk,
                                         boolean addGallery,
                                         String cpuMode);

    public static native boolean nativeRelease(long ctx);
    public static native boolean nativesetAddGallery(long ctx, boolean flag);
    public static native boolean nativeclearGallery(long ctx);
    public static native String nativeProcess(long ctx, Bitmap ARGB888ImageBitmap, String label_name);
    public static native boolean nativesaveIndex(long ctx, String save_file_name);
    public static native boolean nativeloadIndex(long ctx, String load_file_name);
}
