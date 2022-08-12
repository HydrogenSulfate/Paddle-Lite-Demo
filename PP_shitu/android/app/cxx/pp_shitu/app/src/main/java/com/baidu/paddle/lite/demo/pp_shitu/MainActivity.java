package com.baidu.paddle.lite.demo.pp_shitu;

import android.Manifest;
import android.app.ProgressDialog;
import android.content.ContentResolver;
import android.content.Intent;
import android.content.SharedPreferences;
import android.content.pm.PackageManager;
import android.database.Cursor;
import android.graphics.*;
import android.net.Uri;
import android.os.Bundle;
import android.os.Handler;
import android.os.HandlerThread;
import android.os.Message;
import android.preference.PreferenceManager;
import android.provider.MediaStore;
import android.support.annotation.NonNull;
import android.support.v4.app.ActivityCompat;
import android.support.v4.content.ContextCompat;
import android.support.v7.app.AppCompatActivity;
import android.util.Log;
import android.view.*;
import android.widget.*;

import com.baidu.paddle.lite.demo.common.Utils;

import java.io.File;
import java.io.IOException;
import java.io.InputStream;

public class MainActivity extends AppCompatActivity {
    private static final String TAG = MainActivity.class.getSimpleName();
    public static final int OPEN_QUERY_PHOTO_REQUEST_CODE = 0;
    public static final int TAKE_QUERY_PHOTO_REQUEST_CODE = 1;
    public static final int OPEN_GALLERY_PHOTO_REQUEST_CODE = 4;
    public static final int TAKE_GALLERY_PHOTO_REQUEST_CODE = 5;
    public static final int CLEAR_FEATURE_REQUEST_CODE = 6;
    public static final int REQUEST_LOAD_MODEL = 0;
    public static final int REQUEST_RUN_MODEL = 1;
    public static final int RESPONSE_LOAD_MODEL_SUCCESSED = 0;
    public static final int RESPONSE_LOAD_MODEL_FAILED = 1;
    public static final int RESPONSE_RUN_MODEL_SUCCESSED = 2;
    public static final int RESPONSE_RUN_MODEL_FAILED = 3;

    protected ProgressDialog pbLoadModel = null;
    protected ProgressDialog pbRunModel = null;
    protected Handler receiver = null; // Receive messages from worker thread
    protected Handler sender = null; // Send command to worker thread
    protected HandlerThread worker = null; // Worker thread to load&run model

    // UI components of image classification
//    protected TextView tvInputSetting;
    protected ImageView ivInputImage;
    protected TextView tvTop1Result;
    protected TextView tvTop2Result;
    protected TextView tvTop3Result;
    protected TextView tvInferenceTime;
    //protected Switch mSwitch;

    // Model settings of image classification
    protected String modelPath = "";
    protected String labelPath = "";
    protected String indexPath = "";
    protected String imagePath = "";
    protected String DetModelPath = "";
    protected String RecModelPath = "";
    protected int cpuThreadNum = 1;
    protected EditText label_name;
    protected Button label_botton;
    protected boolean add_gallery = false;
    protected int topk = 3;
    protected String cpuMode = "";
    protected long[] detinputShape = new long[]{};
    protected long[] recinputShape = new long[]{};
    protected boolean useGpu = false;
    protected Native predictor = new Native();

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        // Clear all setting items to avoid app crashing due to the incorrect settings
        SharedPreferences sharedPreferences = PreferenceManager.getDefaultSharedPreferences(this);
        SharedPreferences.Editor editor = sharedPreferences.edit();
        editor.clear();
        editor.commit();

        // Prepare the worker thread for mode loading and inference
        receiver = new Handler() {
            @Override
            public void handleMessage(Message msg) {
                switch (msg.what) {
                    case RESPONSE_LOAD_MODEL_SUCCESSED:
                        pbLoadModel.dismiss();
                        onLoadModelSuccessed();
                        break;
                    case RESPONSE_LOAD_MODEL_FAILED:
                        pbLoadModel.dismiss();
                        Toast.makeText(MainActivity.this, "Load model failed!", Toast.LENGTH_SHORT).show();
                        onLoadModelFailed();
                        break;
                    case RESPONSE_RUN_MODEL_SUCCESSED:
                        pbRunModel.dismiss();
                        onRunModelSuccessed();
                        break;
                    case RESPONSE_RUN_MODEL_FAILED:
                        pbRunModel.dismiss();
                        Toast.makeText(MainActivity.this, "Run model failed!", Toast.LENGTH_SHORT).show();
                        onRunModelFailed();
                        break;
                    default:
                        break;
                }
            }
        };
        worker = new HandlerThread("Predictor Worker");
        worker.start();
        sender = new Handler(worker.getLooper()) {
            public void handleMessage(Message msg) {
                switch (msg.what) {
                    case REQUEST_LOAD_MODEL:
                        // Load model and reload test image
                        if (onLoadModel()) {
                            receiver.sendEmptyMessage(RESPONSE_LOAD_MODEL_SUCCESSED);
                        } else {
                            receiver.sendEmptyMessage(RESPONSE_LOAD_MODEL_FAILED);
                        }
                        break;
                    case REQUEST_RUN_MODEL:
                        // Run model if model is loaded
                        if (onRunModel()) {
                            receiver.sendEmptyMessage(RESPONSE_RUN_MODEL_SUCCESSED);
                        } else {
                            receiver.sendEmptyMessage(RESPONSE_RUN_MODEL_FAILED);
                        }
                        break;
                    default:
                        break;
                }
            }
        };

        // Setup the UI components
//        tvInputSetting = findViewById(R.id.tv_input_setting);
        ivInputImage = findViewById(R.id.iv_input_image);
        tvTop1Result = findViewById(R.id.tv_top1_result);
        tvTop2Result = findViewById(R.id.tv_top2_result);
        tvTop3Result = findViewById(R.id.tv_top3_result);
        tvInferenceTime = findViewById(R.id.tv_inference_time);
//        tvInputSetting.setMovementMethod(ScrollingMovementMethod.getInstance());

        // 启动时隐藏输入label的输入框和确定按钮
        label_name = findViewById(R.id.label_name);
        label_name.setVisibility(View.INVISIBLE);
        label_botton = findViewById(R.id.label_botton);
        label_botton.setVisibility(View.INVISIBLE);
    }


    @Override
    protected void onResume() {
        super.onResume();
        SharedPreferences sharedPreferences = PreferenceManager.getDefaultSharedPreferences(this);
        boolean settingsChanged = false;
        String model_path = sharedPreferences.getString(getString(R.string.MODEL_PATH_KEY),
                getString(R.string.MODEL_PATH_DEFAULT));
        String label_path = sharedPreferences.getString(getString(R.string.LABEL_PATH_KEY),
                getString(R.string.LABEL_PATH_DEFAULT));
        String index_path = sharedPreferences.getString(getString(R.string.INDEX_PATH_KEY),
                getString(R.string.INDEX_DIR_DEFAULT));
        String image_path = sharedPreferences.getString(getString(R.string.IMAGE_PATH_KEY),
                getString(R.string.IMAGE_PATH_DEFAULT));
        settingsChanged |= !model_path.equalsIgnoreCase(modelPath);
        settingsChanged |= !label_path.equalsIgnoreCase(labelPath);
        settingsChanged |= !index_path.equalsIgnoreCase(indexPath);
        settingsChanged |= !image_path.equalsIgnoreCase(imagePath);
        int cpu_thread_num = Integer.parseInt(sharedPreferences.getString(getString(R.string.CPU_THREAD_NUM_KEY),
                getString(R.string.CPU_THREAD_NUM_DEFAULT)));
        settingsChanged |= cpu_thread_num != cpuThreadNum;
        long[] det_input_shape =
                Utils.parseLongsFromString(sharedPreferences.getString(getString(R.string.DET_INPUT_SHAPE_KEY),
                        getString(R.string.DET_INPUT_SHAPE_DEFAULT)), ",");
        long[] rec_input_shape =
                Utils.parseLongsFromString(sharedPreferences.getString(getString(R.string.REC_INPUT_SHAPE_KEY),
                        getString(R.string.REC_INPUT_SHAPE_DEFAULT)), ",");
        String cpu_power_mode =
                sharedPreferences.getString(getString(R.string.CPU_POWER_MODE_KEY),
                        getString(R.string.CPU_POWER_MODE_DEFAULT));
        settingsChanged |= !cpu_power_mode.equalsIgnoreCase(cpuMode);
        int top_k = Integer.parseInt(
                sharedPreferences.getString(getString(R.string.INPUT_TOPK_KEY),
                        getString(R.string.INPUT_TOPK_DEFAULT)));
        settingsChanged |= top_k != topk;
        settingsChanged |= det_input_shape.length != detinputShape.length;
        settingsChanged |= rec_input_shape.length != recinputShape.length;
        if (!settingsChanged) {
            for (int i = 0; i < det_input_shape.length; i++) {
                settingsChanged |= det_input_shape[i] != detinputShape[i];
            }
            for (int i = 0; i < rec_input_shape.length; i++) {
                settingsChanged |= rec_input_shape[i] != recinputShape[i];
            }
        }
        if (settingsChanged || useGpu) {
            modelPath = model_path;
            labelPath = label_path;
            indexPath = index_path;
            imagePath = image_path;
            cpuThreadNum = cpu_thread_num;
            detinputShape = det_input_shape;
            recinputShape = rec_input_shape;
            DetModelPath = modelPath;
            RecModelPath = modelPath;
            topk = top_k;
            cpuMode = cpu_power_mode;
            // Update UI
//            tvInputSetting.setText("ModelDir: " + modelPath.substring(modelPath.lastIndexOf("/") + 1) + "\n"
//                    + "CPU" + " Thread Num: " + Integer.toString(cpuThreadNum) + "\n"
//                    + "DetInputShape [" + Long.toString(detinputShape[0]) + "," +
//                    Long.toString(detinputShape[1]) + "," +
//                    Long.toString(detinputShape[2]) + "," +
//                    Long.toString(detinputShape[3]) + "]\n" +
//                    "RecInputShape [" + Long.toString(recinputShape[0]) + "," +
//                    Long.toString(recinputShape[1]) + "," +
//                    Long.toString(recinputShape[2]) + "," +
//                    Long.toString(recinputShape[3]) + "]\n" +
//                    "CPU Mode: " + cpuMode + "\n");
//            tvInputSetting.scrollTo(0, 0);
            // Reload model if configure has been changed
            loadModel();
        }
    }

    public void loadModel() {
        pbLoadModel = ProgressDialog.show(this, "", "Loading model...", false, false);
        sender.sendEmptyMessage(REQUEST_LOAD_MODEL);
    }

    public void runModel() {
        pbRunModel = ProgressDialog.show(this, "", "Running model...", false, false);
        sender.sendEmptyMessage(REQUEST_RUN_MODEL);
    }

//    public void clearFeature() {
//        pbRunModel = ProgressDialog.show(this, "", "Clearing feature...", false, false);
//        sender.sendEmptyMessage(CLEAR_FEATURE_REQUEST_CODE);
//    }

    public boolean onLoadModel() {
        // push model to sdcard
        String realDetModelDir = getExternalFilesDir(null) + "/" + DetModelPath;
        Utils.copyDirectoryFromAssets(this, DetModelPath, realDetModelDir);
        String realRecModelDir = getExternalFilesDir(null) + "/" + RecModelPath;
        Utils.copyDirectoryFromAssets(this, RecModelPath, realRecModelDir);

        // push label to sdcard
        String realLabelPath = getExternalFilesDir(null) + "/" + labelPath;
        Utils.copyFileFromAssets(this, labelPath, realLabelPath);
        String realIndexDir = getExternalFilesDir(null) + "/" + indexPath;
        Utils.copyFileFromAssets(this, indexPath, realIndexDir);

        return predictor.init(realDetModelDir, realRecModelDir, realLabelPath, realIndexDir,
                detinputShape, recinputShape, cpuThreadNum, 0, 1, topk, add_gallery, cpuMode);
    }

    public boolean onRunModel() {
        return predictor.isLoaded() && predictor.process();
    }

    public void onLoadModelSuccessed() {
        // Load test image from path and run model
        try {
            if (imagePath.isEmpty()) {
                return;
            }
            Bitmap image = null;
            // Read test image file from custom path if the first character of mode path is '/', otherwise read test
            // image file from assets
            if (!imagePath.substring(0, 1).equals("/")) {
                InputStream imageStream = getAssets().open(imagePath);
                image = BitmapFactory.decodeStream(imageStream);
            } else {
                if (!new File(imagePath).exists()) {
                    return;
                }
                image = BitmapFactory.decodeFile(imagePath);
            }
            if (image != null && predictor.isLoaded()) {
                predictor.setInputImage(image);
                runModel();
            }
        } catch (IOException e) {
            Toast.makeText(MainActivity.this, "Load image failed!", Toast.LENGTH_SHORT).show();
            e.printStackTrace();
        }
    }

    public void onLoadModelFailed() {
    }

    public void onRunModelSuccessed() {
        // Obtain results and update UI
        tvInferenceTime.setText("Inference time: " + predictor.inferenceTime() + " ms");
        Bitmap inputImage = predictor.inputImage();
        if (inputImage != null) {
            ivInputImage.setImageBitmap(inputImage);
        }
        tvTop1Result.setText(predictor.top1Result());
        tvTop2Result.setText(predictor.top2Result());
        tvTop3Result.setText(predictor.top3Result());
    }

    public void onRunModelFailed() {
    }

    public void onImageChanged(Bitmap image) {
        // Rerun model if users pick test image from gallery or camera
        if (image != null && predictor.isLoaded()) {
            label_name.setVisibility(View.INVISIBLE);
            label_botton.setVisibility(View.INVISIBLE);
            predictor.setAddGallery(false);
            predictor.setInputImage(image);
            runModel();
        }
    }

    public void onAddGallery(Bitmap image) {
        if (image != null && predictor.isLoaded()) {
            label_name.setVisibility(View.VISIBLE);
            label_botton.setVisibility(View.VISIBLE);
            label_name.setHint("image label name");
            ivInputImage.setImageBitmap(image);
            label_botton.setOnClickListener(new View.OnClickListener() {
                @Override
                public void onClick(View view) {
                    predictor.setAddGallery(true);
                    predictor.setLabelName(label_name.getText().toString());
                    predictor.setInputImage(image);
                    runModel();
                    label_name.setVisibility(View.INVISIBLE);
                    label_botton.setVisibility(View.INVISIBLE);
                    label_name.setText("");
                }
            });
        }
    }

//    public void onClearFeature() {
//        predictor.clearFeature();
//    }

    public void onSettingsClicked() {
        startActivity(new Intent(MainActivity.this, SettingsActivity.class));
    }

    @Override
    public boolean onCreateOptionsMenu(Menu menu) {
        MenuInflater inflater = getMenuInflater();
        inflater.inflate(R.menu.menu_action_options, menu);

        MenuItem take_query = menu.findItem(R.id.take_query);
        MenuItem add_query = menu.findItem(R.id.add_query);
        MenuItem take_gallery = menu.findItem(R.id.take_gallery);
        MenuItem add_gallery = menu.findItem(R.id.add_gallery);

        take_query.setOnMenuItemClickListener(new MenuItem.OnMenuItemClickListener() {
            @Override
            public boolean onMenuItemClick(MenuItem menuItem) {
                takeQueryPhoto();
                return true;
            }
        });
        add_query.setOnMenuItemClickListener(new MenuItem.OnMenuItemClickListener() {
            @Override
            public boolean onMenuItemClick(MenuItem menuItem) {
                openQueryPhoto();
                return true;
            }
        });
        take_gallery.setOnMenuItemClickListener(new MenuItem.OnMenuItemClickListener() {
            @Override
            public boolean onMenuItemClick(MenuItem menuItem) {
                takeGalleryPhoto();
                return true;
            }
        });
        add_gallery.setOnMenuItemClickListener(new MenuItem.OnMenuItemClickListener() {
            @Override
            public boolean onMenuItemClick(MenuItem menuItem) {
                openGalleryPhoto();
                return true;
            }
        });
        return true;
    }


    public boolean onPrepareOptionsMenu(Menu menu) {
        boolean isLoaded = predictor.isLoaded();
//        menu.findItem(R.id.open_gallery).setEnabled(isLoaded);
//        menu.findItem(R.id.take_photo).setEnabled(isLoaded);
//        menu.findItem(R.id.add_exist_photo).setEnabled(isLoaded);
//        menu.findItem(R.id.add_token_photo).setEnabled(isLoaded);
//        menu.findItem(R.id.clear_gallery).setEnabled(isLoaded);
//        menu.findItem(R.id.save_index).setEnabled(isLoaded);
//        menu.findItem(R.id.load_index).setEnabled(isLoaded);
        return super.onPrepareOptionsMenu(menu);
    }

//    public void showPopup(View view)
//    {
//        PopupMenu popupMenu = new PopupMenu(MainActivity.this, view);
//        popupMenu.getMenuInflater().inflate(R.menu.menu_action_options, popupMenu.getMenu());
//        popupMenu.show();
//    }
    @Override
    public boolean onOptionsItemSelected(MenuItem item) {
        switch (item.getItemId()) {
            case android.R.id.home:
                finish();
                break;
            case R.id.add_query:
                if (requestAllPermissions()) {
                    openQueryPhoto();
                }
                break;
            case R.id.take_query:
                if (requestAllPermissions()) {
                    takeQueryPhoto();
                }
                break;
//            case R.id.clear_gallery:
//                if (requestAllPermissions()) {
//                    clearIndex();
//                }
//                break;
            case R.id.settings:
                if (requestAllPermissions()) {
                    // Make sure we have SDCard r&w permissions to load model from SDCard
                    onSettingsClicked();
                }
                break;
            case R.id.add_gallery:
                if (requestAllPermissions()) {
                    // 从本地选择一张图片放入gallery中
                    openGalleryPhoto();
                }
                break;
            case R.id.take_gallery:
                if (requestAllPermissions()) {
                    // 从本地选择一张图片放入gallery中
                    takeGalleryPhoto();
                }
                break;
//            case R.id.save_index:
//                if (requestAllPermissions()) {
//                    // 从本地选择一张图片放入gallery中
//                    saveIndex();
//                }
//                break;
//            case R.id.load_index:
//                if (requestAllPermissions()) {
//                    // 从本地选择一张图片放入gallery中
//                    loadIndex();
//                }
//                break;
        }
        return super.onOptionsItemSelected(item);
    }


    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions,
                                           @NonNull int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        if (grantResults[0] != PackageManager.PERMISSION_GRANTED || grantResults[1] != PackageManager.PERMISSION_GRANTED) {
            Toast.makeText(this, "Permission Denied", Toast.LENGTH_SHORT).show();
        }
    }

    private boolean requestAllPermissions() {
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.WRITE_EXTERNAL_STORAGE)
                != PackageManager.PERMISSION_GRANTED || ContextCompat.checkSelfPermission(this,
                Manifest.permission.CAMERA)
                != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(this, new String[]{Manifest.permission.WRITE_EXTERNAL_STORAGE,
                            Manifest.permission.CAMERA},
                    0);
            return false;
        }
        return true;
    }

    private void openQueryPhoto() {
        Intent intent = new Intent(Intent.ACTION_PICK, null); // 选择数据
        intent.setDataAndType(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, "image/*");
        startActivityForResult(intent, OPEN_QUERY_PHOTO_REQUEST_CODE);
    }

    private void takeQueryPhoto() {
        Intent takePhotoIntent = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
        if (takePhotoIntent.resolveActivity(getPackageManager()) != null) {
            startActivityForResult(takePhotoIntent, TAKE_QUERY_PHOTO_REQUEST_CODE);
        }
    }

    private void openGalleryPhoto() {
        // 直接拍一张图片到库中---主逻辑代码
        Intent intent = new Intent(Intent.ACTION_PICK, null); // 选择数据
        intent.setDataAndType(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, "image/*");
        startActivityForResult(intent, OPEN_GALLERY_PHOTO_REQUEST_CODE);
        Toast.makeText(MainActivity.this, "AddTokenToGallery", Toast.LENGTH_SHORT).show();
    }

    private void takeGalleryPhoto() {
        // 增加现有图片到库中---主逻辑代码
        Intent intent = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
        if (intent.resolveActivity(getPackageManager()) != null) {
            startActivityForResult(intent, TAKE_GALLERY_PHOTO_REQUEST_CODE);
        }
        Toast.makeText(MainActivity.this, "AddExistToGallery", Toast.LENGTH_SHORT).show();
    }

    private void clearIndex() {
        // 清空index主逻辑代码
        predictor.clearFeature();
        Toast.makeText(MainActivity.this, "clearIndex", Toast.LENGTH_SHORT).show();
    }

    private void saveIndex() {
        // 保存index（以及同名的id_map)主逻辑代码
        label_name.setVisibility(View.VISIBLE);
        label_botton.setVisibility(View.VISIBLE);
        label_name.setHint("save index file name");
        label_botton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                predictor.saveIndex(label_name.getText().toString());
                label_name.setVisibility(View.INVISIBLE);
                label_botton.setVisibility(View.INVISIBLE);
                label_name.setText("");
                Toast.makeText(MainActivity.this, "index saved", Toast.LENGTH_SHORT).show();
            }
        });
    }

    private void loadIndex() {
        // 载入index（以及同名的id_map)主逻辑代码
        label_name.setVisibility(View.VISIBLE);
        label_botton.setVisibility(View.VISIBLE);
        label_name.setHint("load index file name");
        label_botton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                boolean flag = predictor.loadIndex(label_name.getText().toString());
                label_name.setVisibility(View.INVISIBLE);
                label_botton.setVisibility(View.INVISIBLE);
                label_name.setText("");
                if (flag) {
                    Toast.makeText(MainActivity.this, "index loaded success", Toast.LENGTH_SHORT).show();
                } else {
                    Toast.makeText(MainActivity.this, "index loaded failed, check index filename", Toast.LENGTH_SHORT).show();
                }
            }
        });
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if (resultCode == RESULT_OK && data != null) {
            switch (requestCode) {
                case OPEN_QUERY_PHOTO_REQUEST_CODE:
                    try {
                        ContentResolver resolver = getContentResolver();
                        Uri uri = data.getData();
                        Bitmap image = MediaStore.Images.Media.getBitmap(resolver, uri);
                        String[] proj = {MediaStore.Images.Media.DATA};
                        Cursor cursor = managedQuery(uri, proj, null, null, null);
                        cursor.moveToFirst();
                        onImageChanged(image);
                    } catch (IOException e) {
                        Log.e(TAG, e.toString());
                    }
                    break;
                case OPEN_GALLERY_PHOTO_REQUEST_CODE:
                    try {

                        ContentResolver resolver = getContentResolver();
                        Uri uri = data.getData();
                        Bitmap image = MediaStore.Images.Media.getBitmap(resolver, uri);
                        String[] proj = {MediaStore.Images.Media.DATA};
                        Cursor cursor = managedQuery(uri, proj, null, null, null);
                        cursor.moveToFirst();
                        onAddGallery(image);

                    } catch (IOException e) {
                        Log.e(TAG, e.toString());
                    }
                    break;
                case TAKE_GALLERY_PHOTO_REQUEST_CODE:
                    Bundle gextras = data.getExtras();
                    Bitmap gimage = (Bitmap) gextras.get("data");
                    onAddGallery(gimage);
                    break;
                case TAKE_QUERY_PHOTO_REQUEST_CODE:
                    Bundle extras = data.getExtras();
                    Bitmap image = (Bitmap) extras.get("data");
                    onImageChanged(image);
                    break;
                case CLEAR_FEATURE_REQUEST_CODE:
                    clearIndex();
                    break;
                default:
                    break;
            }
        }
    }

    @Override
    protected void onDestroy() {
        if (predictor != null) {
            predictor.release();
        }
        worker.quit();
        super.onDestroy();
    }
}
