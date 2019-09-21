import tensorflow as tf

with tf.gfile.FastGFile('output_graph.pb', "rb") as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    del(graph_def.node[-1])
    
    g_in = tf.import_graph_def(graph_def, name="")
    flops = tf.profiler.profile(g_in, options = tf.profiler.ProfileOptionBuilder.float_operation())
    print('FLOP after freezing', flops.total_float_ops)
