import os
import sys
from random import randint
import tensorflow as tf
import hw_utils
import pandas as pd
import ANN_model
import json
import ast
import numpy as np

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
        print(">> python3 test.py [path_config_file]")
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



    im_path=str(config["general"]["processed_data_path"])
    csv_path=str(config["IAM-test"]["csv_path"])
    results_path=str(config["IAM-test"]["results_path"])
    checkpoints_path=str(config["IAM-test"]["checkpoints_path"])
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
    cost=model[6]
    ler=model[7]
    decoded=model[8]

    result_test = pd.DataFrame()
    config = tf.ConfigProto()
    config.gpu_options.allocator_type = 'BFC'


    with tf.Session(graph=graph,config = config) as session:

    
        saver = tf.train.Saver()
        saver.restore(session, "C:/Users/catarina/QRCodeDetection/Offline-Handwriting-Recognition-with-TensorFlow/Checkpoints/checkpoint_epoch_27400_ler_0.15915638256412773.ckpt") #hardcodded change it
        print("Loaded Model")


        cont = 0
        total_test_cost = 0
        total_test_ler = 0

   
        while cont >= 0:

            test_inputs, test_targets, original, test_seq_len, num_samples = hw_utils.extract_ordered_batch(
                ctc_input_len, 1, im_path, csv_path + "test_tickets.csv", cont)

            if num_samples==1:

                test_feed = {seq_len: test_seq_len,
                             inputs: test_inputs,
                             keep_prob: 1,
                             targets:test_targets}
                test_cost, test_ler= session.run([cost, ler], test_feed)
                total_test_cost += test_cost
                total_test_ler += test_ler

                dec=session.run(decoded[0],test_feed)
                output = str(list(map(dct.get, list(dec.values))))
                for ch in ["['", "']", "', '"]:
                    output = output.replace(ch, "")
                    original=str(original).replace(ch, "")
                tuple = {'Target': [original], 'Output': [output]}
                result_test = pd.concat([result_test, pd.DataFrame(tuple)])
                cont += 1

            else:

                print("IAM test result:")
                print("Cost: "+str(total_test_cost / (cont)))
                print("LER: "+str(total_test_ler / (cont)))
                result_test.to_csv(results_path+"test_result_tickets.csv",index=False) 
                cont = -1

if __name__ == '__main__':
    run_ctc()
