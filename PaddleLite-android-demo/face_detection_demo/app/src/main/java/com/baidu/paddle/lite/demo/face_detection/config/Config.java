package com.baidu.paddle.lite.demo.face_detection.config;

import android.graphics.Bitmap;

public class Config {

    public String modelPath = "";
    public String labelPath = "";
    public String imagePath = "";
    public int cpuThreadNum = 1;
    public String cpuPowerMode = "";
    public String inputColorFormat = "";
    public long[] inputShape = new long[]{};
    public float[] inputMean = new float[]{};
    public float[] inputStd = new float[]{};

    public void init(String modelPath, String labelPath, String imagePath, int cpuThreadNum,
                     String cpuPowerMode, String inputColorFormat,long[] inputShape,
                     float[] inputMean,float[] inputStd ){

        this.modelPath = modelPath;
        this.labelPath = labelPath;
        this.imagePath = imagePath;
        this.cpuThreadNum = cpuThreadNum;
        this.cpuPowerMode = cpuPowerMode;
        this.inputColorFormat = inputColorFormat;
        this.inputShape = inputShape;
        this.inputMean = inputMean;
        this.inputStd = inputStd;
    }

    public void setInputShape(Bitmap inputImage){
        this.inputShape[0] = 1;
        this.inputShape[1] = 3;
        this.inputShape[2] = inputImage.getHeight();
        this.inputShape[3] = inputImage.getWidth();

    }

}
