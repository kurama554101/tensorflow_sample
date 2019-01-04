import onnx_tf.backend
import onnx
import numpy as np
import os
import tensorflow as tf
import time

from tensorflow.contrib.compiler import xla

model_root_path = "models"
model_path = os.path.join(model_root_path, "resnet50v2.onnx")


def download(url, path, overwrite=False):
    import os
    if os.path.isfile(path) and not overwrite:
        print('File {} existed, skip.'.format(path))
        return
    print('Downloading from url {} to {}'.format(url, path))
    try:
        import urllib.request
        urllib.request.urlretrieve(url, path)
    except:
        import urllib
        urllib.urlretrieve(url, path)


def main():
    print("## start ##")
    if not os.path.exists(model_root_path):
        print("## create model folder ##")
        os.mkdir(model_root_path)

    if not os.path.exists(model_path):
        print("## download model ##")
        download("https://s3.amazonaws.com/onnx-model-zoo/resnet/resnet50v2/resnet50v2.onnx", model_path, False)

    print("## load onnx model ##")
    onnx_model = onnx.load_model(model_path)

    print("## convert onnx -> tf_graph ##")
    tf_model = onnx_tf.backend.prepare(onnx_model, device="CPU")
    tf_graph = tf_model.graph
    placeholders = tf.contrib.framework.get_placeholders(tf_graph)

    print("## prepare input ##")
    x = np.reshape(np.arange(1 * 3 * 224 * 224, dtype="float32") * 0.5, (1, 3, 224, 224))

    print("## prepare tf model ##")
    y = tf_graph.get_tensor_by_name("add_1:0")  # for ResNet50
    feed_dict = {placeholders[0]: x}

    def create_graph(xx):
        return [y]

    print("## start session ##")
    config = tf.ConfigProto()
    config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1
    with tf.device("device:XLA_CPU:0"):
        with tf.Session(graph=tf_graph, config=config) as sess:
            print("## compile xla ##")
            y_ = xla.compile(create_graph, [x])  # TODO:xla.compileしても推論速度が変わっていないため、使い方が間違っていそう・・。

            print("## start run ##")
            start_time = time.time()
            out = sess.run(y_, feed_dict=feed_dict)
            exec_time = time.time() - start_time
            print(out)
            print(exec_time)


def debug_outputs(tf_model):
    for output in tf_model.outputs:
        print(output)


def debug_placeholder(tf_placeholders):
    for placeholder in tf_placeholders:
        print(placeholder)


def debug_tensor(tf_graph):
    for op in tf_graph.get_operations():
        print(op.outputs)


if __name__ == "__main__":
    main()
