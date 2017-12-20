import tensorflow as tf

def CNN_RNN_CTC(kernel_size, num_conv1, num_conv2, num_conv3, num_conv4, num_conv5, num_rnn, num_fc, HEIGHT, WIDTH, num_classes):
    """
               args:

          - kernel_size: kernel size for CNN. (Int)
          - num_conv1: 1st CNN size. (Int)
          - num_conv2: 2nd CNN size. (Int)
          - num_conv3: 3rd CNN size. (Int)
          - num_conv4: 4th CNN size. (Int)
          - num_conv5: 5th CNN size. (Int)
          - num_rnn: RNN size. (Int)
          - num_fc:Fullyconnected size. (Int)
          - HEIGHT: of images (Int)
          - WIDTH: of images (Int)
          - num_classes: number of classes plus one for unknown. (Int)

        out:

          - graph: (Graph)
          - inputs: Placeholder for input (Placeholder)
          - targets: Placeholder for output (Placeholder)
          - keep_prob: Placeholder for dropout prob. (Placeholder)
          - seq_len: Placeholder for input entry in CTC . (Placeholder)
          - optimizer:  (Operation)
          - cost: model error (Operation)
          - ler: label error rate (Operation)
          - decoded: decode the output (Operation)
    """


    graph = tf.Graph()
    with graph.as_default():

        # PLACEHOLDERS
        # input: Tensor [batch_size, height, width, channel] 
        inputs = tf.placeholder(tf.float32, [None, HEIGHT, WIDTH,1])

        # Tensor [batch_size_allongtheway]
        seq_len = tf.placeholder(tf.int32, [None])


        keep_prob = tf.placeholder(tf.float32)


        targets = tf.sparse_placeholder(tf.int32)


        # VARIABLES
        # filters and bias of CNNs
        w_conv1 = tf.Variable(tf.random_normal([kernel_size, kernel_size, 1,num_conv1], stddev=0.01))
        b_conv1 = tf.Variable(tf.random_normal([num_conv1], stddev=0.01))
        w_conv2 = tf.Variable(tf.random_normal([kernel_size, kernel_size, num_conv1, num_conv2], stddev=0.01))
        b_conv2 = tf.Variable(tf.random_normal([num_conv2], stddev=0.01))
        w_conv3 = tf.Variable(tf.random_normal([kernel_size, kernel_size, num_conv2,num_conv3], stddev=0.01))
        b_conv3 = tf.Variable(tf.random_normal([num_conv3], stddev=0.01))
        w_conv4 = tf.Variable(tf.random_normal([kernel_size, kernel_size, num_conv3, num_conv4], stddev=0.01))
        b_conv4 = tf.Variable(tf.random_normal([num_conv4], stddev=0.01))
        w_conv5 = tf.Variable(tf.random_normal([kernel_size, kernel_size, num_conv4, num_conv5], stddev=0.01))
        b_conv5 = tf.Variable(tf.random_normal([num_conv5], stddev=0.01))

        # weights RNN
        lstm_fw_cell = tf.contrib.rnn.BasicLSTMCell(num_rnn, forget_bias=1.0)
        lstm_bw_cell = tf.contrib.rnn.BasicLSTMCell(num_rnn, forget_bias=1.0)

        # weights FULLCONNECT
        w1 = tf.Variable(tf.random_normal([2*num_rnn, num_fc], stddev=0.01))
        b1 = tf.Variable(tf.random_normal([num_fc], stddev=0.01))
        w2 = tf.Variable(tf.random_normal([num_fc,num_classes], stddev=0.01))
        b2 = tf.Variable(tf.random_normal([num_classes], stddev=0.01))



        # ARCHITECTURE
        # Normalization
        inputs = tf.nn.l2_normalize(inputs, [1, 2])

        #LAYER 1 CNN-MAXPOOL-DROP
        h_conv1 = tf.nn.relu(tf.nn.conv2d(inputs, w_conv1, strides=[1,1,1,1], padding='SAME') + b_conv1)
        h_pool1 = tf.nn.max_pool(h_conv1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
        h_pool1=tf.nn.l2_normalize(h_pool1,[1,2])
        h_pool1=tf.nn.dropout(h_pool1,keep_prob=keep_prob)

        # LAYER 2 CNN-MAXPOOL-DROP
        h_conv2 = tf.nn.relu(tf.nn.conv2d(h_pool1, w_conv2, strides=[1,1,1,1], padding='SAME') + b_conv2)
        h_pool2 = tf.nn.max_pool(h_conv2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
        h_pool2=tf.nn.l2_normalize(h_pool2,[1,2])
        h_pool2=tf.nn.dropout(h_pool2,keep_prob=keep_prob)

        # LAYER 3 CNN-MAXPOOL-DROP
        h_conv3 = tf.nn.relu(tf.nn.conv2d(h_pool2, w_conv3, strides=[1,1,1,1], padding='SAME') + b_conv3)
        h_pool3 = tf.nn.max_pool(h_conv3, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
        h_pool3=tf.nn.l2_normalize(h_pool3,[1,2])
        h_pool3=tf.nn.dropout(h_pool3,keep_prob=keep_prob)

        # LAYER 4 CNN-MAXPOOL-DROP
        h_conv4 = tf.nn.relu(tf.nn.conv2d(h_pool3, w_conv4, strides=[1,1,1,1], padding='SAME') + b_conv4)
        h_pool4 = tf.nn.max_pool(h_conv4, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
        h_pool4=tf.nn.l2_normalize(h_pool4,[1,2])
        h_pool4=tf.nn.dropout(h_pool4,keep_prob=keep_prob)

        # LAYER 5 CNN-MAXPOOL-DROP
        h_conv5 = tf.nn.relu(tf.nn.conv2d(h_pool4, w_conv5, strides=[1,1,1,1], padding='SAME') + b_conv5)
        h_pool5 = tf.nn.max_pool(h_conv5, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
        h_pool5=tf.nn.l2_normalize(h_pool5,[1,2])
        h_pool5=tf.nn.dropout(h_pool5,keep_prob=keep_prob)

        # LAYER 6 RNNs
        outputs=tf.transpose(h_pool5, (2,0,1,3))
        outputs=tf.reshape(outputs, (int(WIDTH/(2**5)),-1,int(HEIGHT*num_conv5/(2**5))))
        outputs=tf.transpose(outputs, (1,0,2))
        outputs, _= tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell, lstm_bw_cell, outputs,dtype=tf.float32)
        outputs=tf.concat(outputs,2)
        outputs=tf.transpose(outputs, (1,0,2))        
        outputs=tf.reshape(outputs, (-1,2*num_rnn))

        # LAYER 7 FULLCONNECT-CTC_Loss
        logits = tf.matmul(outputs, w1) + b1
        logits =  tf.matmul(logits, w2) + b2
        logits = tf.reshape(logits, (int(WIDTH/(2**5)),-1,num_classes))
        loss = tf.nn.ctc_loss(targets, logits, seq_len,preprocess_collapse_repeated=True)
        cost = tf.reduce_mean(loss)



        # minimizing the error
        optimizer = tf.train.AdamOptimizer(learning_rate=0.01).minimize(cost)


        # decode to extract the sequence of charcaters
        decoded, log_prob = tf.nn.ctc_greedy_decoder(logits, seq_len)

        # Error: Label Error Rate
        ler = tf.reduce_mean(tf.edit_distance(tf.cast(decoded[0], tf.int32), targets))

    return graph, inputs, targets, keep_prob, seq_len, optimizer, cost, ler, decoded
