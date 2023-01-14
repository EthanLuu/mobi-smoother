import tensorflow as tf
import numpy as np
import cv2
import os

root_path = os.getcwd()
model_path = f'{root_path}/vsr/ckp'
vsr_model_path = f'{model_path}/model.tflite'
image_root_path = f"{root_path}/data/image"

read_1 = cv2.imread(f"{root_path}/data/.png")
read_1 = tf.convert_to_tensor(read_1)
read_1 = tf.cast(read_1, tf.float32)

read_1 = read_1 / 255.  # type: ignore # 0~1

# 10 frames
inputs = tf.concat([read_1 for i in range(10)], axis=-1)
inputs = tf.expand_dims(inputs, axis=0)
print(inputs.shape)

interpreter = tf.lite.Interpreter(model_path=vsr_model_path)

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
print(input_details, '\n', output_details)

interpreter.resize_tensor_input(input_details[0]['index'], [1, 180, 320, 30],
                                strict=False)
interpreter.allocate_tensors()
interpreter.set_tensor(input_details[0]['index'], inputs)
interpreter.invoke()

output_data = interpreter.get_tensor(output_details[0]['index'])

print(type(output_data))

output1 = np.clip(output_data[0, :, :, 0:3], a_min=0, a_max=1)

output1 = (output1 * 255).astype(np.uint8)

cv2.imwrite("res2.png", output1)
