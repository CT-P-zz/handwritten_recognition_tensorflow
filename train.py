
import os
import sys
from random import randint
import tensorflow as tf
import hw_utils
import pandas as pd
import ANN_model
import json
import ast

def run_ctc():

    if len(sys.argv) == 1:
        print("Execution without arguments, config file by default: ./config.json")
        config_file = str('./config.json')

    elif len(sys.argv) == 2:
        print("Execution with arguments, config file:" + str(sys.argv[1]))
        config_file = str(sys.argv[1])

    else:
        print()
        print("ERROR")
        print("Wrong number of arguments. Execute:")
        print(">> python3 train.py [path_config_file]")
        exit(1)

  
    try:
        config = json.load(open(config_file))
    except FileNotFoundError:
        print()
        print("ERROR")
        print("No such config file : " + config_file)
        exit(1)

    if not os.path.exists(str(config["IAM-test"]["results_path"])):
        os.mkdir(str(config["IAM-test"]["results_path"]))
    if not os.path.exists(str(config["IAM-test"]["checkpoints_path"])):
        os.mkdir(str(config["IAM-test"]["checkpoints_path"]))



    im_path=str(config["general"]["processed_data_path"])
    csv_path=str(config["IAM-test"]["csv_path"])
    results_path=str(config["IAM-test"]["results_path"])
    checkpoints_path=str(config["IAM-test"]["checkpoints_path"])
    batch_size = int(config["IAM-test"]["batch_size"])
    num_epochs = int(config["IAM-test"]["num_epochs"])
    val_period = int(config["IAM-test"]["validation_period"])
    print_period = int(config["IAM-test"]["print_period"])
    height = int(config["general"]["height"])
    width = int(config["general"]["width"])
    dct=ast.literal_eval(str(config["general"]["dictionary"]))


    kernel_size=int(config["cnn-rnn-ctc"]["kernel_size"])
    num_conv1=int(config["cnn-rnn-ctc"]["num_conv1"])
    num_conv2=int(config["cnn-rnn-ctc"]["num_conv2"])
    num_conv3=int(config["cnn-rnn-ctc"]["num_conv3"])
    num_conv4=int(config["cnn-rnn-ctc"]["num_conv4"])
    num_conv5=int(config["cnn-rnn-ctc"]["num_conv5"])
    num_rnn=int(config["cnn-rnn-ctc"]["num_rnn"])
    num_fc=int(config["cnn-rnn-ctc"]["num_fc"])
    num_classes=int(config["cnn-rnn-ctc"]["num_classes"])
    ctc_input_len=int(config["cnn-rnn-ctc"]["ctc_input_len"])

    model = ANN_model.CNN_RNN_CTC(kernel_size, num_conv1, num_conv2, num_conv3, num_conv4,
                               num_conv5, num_rnn, num_fc, height, width, num_classes)
    graph=model[0]
    inputs=model[1]
    targets=model[2]
    keep_prob=model[3]
    seq_len=model[4]
    optimizer=model[5]
    cost=model[6]
    ler=model[7]
    decoded=model[8]

    train_result = pd.DataFrame()
    val_result1 = pd.DataFrame()
    val_result2 = pd.DataFrame()

    


    with tf.Session(graph=graph) as session:

        saver=tf.train.Saver()
        tf.global_variables_initializer().run()


        LER=1.0

        for curr_epoch in range(num_epochs):

       
            train_inputs, train_targets, original, train_seq_len = hw_utils.extract_training_batch(ctc_input_len,batch_size,im_path,csv_path + "train.csv")
            feed = {inputs: train_inputs, targets: train_targets, keep_prob: 0.5, seq_len: train_seq_len}

 
            _ = session.run([optimizer], feed)

  
            if curr_epoch % val_period == 0:
   
                train_cost, train_ler = session.run([cost, ler], feed)
                train_tuple = {'epoch': [curr_epoch], 'train_cost': [train_cost], 'train_ler': [train_ler]}
                train_result = pd.concat([train_result, pd.DataFrame(train_tuple)])

                val_tuple1=hw_utils.validation(curr_epoch,ctc_input_len, batch_size, im_path, csv_path + "validation1.csv", inputs, targets, keep_prob, seq_len, session,
                           cost, ler)
                val_result1 = pd.concat([val_result1, pd.DataFrame(val_tuple1)])

                val_tuple2 = hw_utils.validation(curr_epoch,ctc_input_len, batch_size, im_path, csv_path + "validation2.csv", inputs,
                                           targets, keep_prob, seq_len, session,
                                           cost, ler)
                val_result2 = pd.concat([val_result2, pd.DataFrame(val_tuple2)])


                if (float(val_tuple1['val_ler'][0])+float(val_tuple2['val_ler'][0]))/2 <= LER:

                    save_path = saver.save(session, checkpoints_path+"checkpoint_epoch_"+str(curr_epoch)+"_ler_"+str((float(val_tuple1['val_ler'][0])+float(val_tuple2['val_ler'][0]))/2)+".ckpt")
                    global_step = ckpt.checkpoints_path.split('/')[-1].split('data')[-1]
                    print("Model saved in file: " +str(save_path))

   
                    LER=(float(val_tuple1['val_ler'][0])+float(val_tuple2['val_ler'][0]))/2


                if curr_epoch % print_period == 0:


                    print("Epoch: "+ str(curr_epoch) + " val_cost: " + str((float(val_tuple1['val_cost'][0])+float(val_tuple2['val_cost'][0]))/2) + " val_ler: " + str((float(val_tuple1['val_ler'][0])+float(val_tuple2['val_ler'][0]))/2))

                 
                    print("Examples:")
                    for j in range(10):

                        prob_inputs, prob_targets, prob_original, prob_seq_len, _ = hw_utils.extract_ordered_batch(ctc_input_len,1,im_path,csv_path + "validation1.csv",randint(0,6086))
                        prob_feed = {inputs: prob_inputs,
                                     targets: prob_targets,
                                     keep_prob: 1,
                                     seq_len: prob_seq_len}

                   
                        prob_d = session.run(decoded[0], feed_dict=prob_feed)
                        output = str(list(map(dct.get, list(prob_d.values))))
                        for ch in ["['", "']", "', '"]:
                            output = output.replace(ch, "")
                            prob_original=str(prob_original).replace(ch, "")
                        print("Target: " + prob_original +"       Model Output: " + output)

        val_result1.to_csv(results_path+"validation_result1.csv",index=False)
        val_result2.to_csv(results_path+"validation_result2.csv",index=False)
        train_result.to_csv(results_path+"training_result.csv",index=False)
        print("THE TRAINING IS OVER")


if __name__ == '__main__':
    run_ctc()
