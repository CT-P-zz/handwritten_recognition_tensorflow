import os
import sys
from PIL import Image
import pandas as pd
import numpy as np


def scale_invert(raw_path, proc_path,height,width):

    """
    Rescale images and invert the colors to binary. Adds background, so the images have all the same height and width

    Args:

      - raw_path: original path of the images (String)
      - proc_path: path of the processed iamges (String)
      - height:  (Int)
      - width: (Int)

    """
    
    im = Image.open(raw_path)
    
    # rescale
    raw_width, raw_height = im.size
    new_width = int(round(raw_width * (height / raw_height)))
    im = im.resize((new_width, height), Image.NEAREST)
    im_map = list(im.getdata())
    im_map = np.array(im_map)
    im_map = im_map.reshape(height, new_width).astype(np.uint8)

    # Invert and add background (black - 255) 
    data = np.full((height, width - new_width + 1), 255)
    im_map = np.concatenate((im_map, data), axis=1)
    im_map = im_map[:, 0:width]
    im_map = (255 - im_map)
    im_map = im_map.astype(np.uint8)
    im = Image.fromarray(im_map)

    
    im.save(str(proc_path), "png")
    print("Processed image saved: " + str(proc_path))


def extract_training_batch(ctc_input_len,batch_size,im_path,csv_path):

    """
        Batch of images and labels in a random way to train the ANN

        Args:

          - ctc_input_len: input length of ctc (int)
          - batch_size:  (Int)
          - im_path: image path (String)
          - csv_path: training dataset path (Int)

        out:

          - batchx: Tensor with images as matrices for ANN input.
            (Array of Floats: [batch_size, height, width, 1])
          - sparse: SparseTensor with labels (SparseTensor: indice,values,shape)
          - transcriptions: Arraywith labels of "batchx". (Array de Strings: [batch_size])
          - seq_len: Array with input length for CTC, "ctc_input_len". (Array of Ints: [batch_size])
    """

    
    df = pd.read_csv(csv_path, sep=",",index_col="index")
    df_sample=df.sample(batch_size).reset_index()


    batchx = []
    transcriptions = []
    index = []
    values=[]
    seq_len=[]

    
    for i in range(batch_size):
        im_apt = df_sample.loc[i, ['image']].as_matrix()
        df_y =df_sample.loc[i, ['transcription']].as_matrix()
        for fich in im_apt:

          
            fich = str(fich)
            fich = fich.replace("['", "").replace("']", "")
            im = Image.open(im_path + fich + ".png")
            width, height = im.size
            im_map = list(im.getdata())
            im_map = np.array(im_map)
            im_map = im_map / 255
            result=im_map.reshape(height, width,1)
            batchx.append(result)

            # extract labels
            original=""
            for n in list(str(df_y)):
                if n == n.lower() and n == n.upper():
                    if n in "0123456789":
                        values.append(int(n))
                        original = original + n
                elif n == n.lower():
                    values.append(int(ord(n) - 61))
                    original = original + n
                elif n == n.upper():
                    values.append(int((ord(n) - 55)))
                    original = original + n

         
            for j in range(len(str(df_y))-4):
                index.append([i,j])

         
            transcriptions.append(original)
            seq_len.append(ctc_input_len)

    # normalization
    batchx = np.stack(batchx, axis=0)

    
    shape=[batch_size,18]
    sparse=index,values,shape

    return batchx, sparse, transcriptions, seq_len


def extract_ordered_batch(ctc_input_len,batch_size,im_path,csv_path,cont):

    """
       Batch of images and labels in a sequential way to train the ANN

        Args:
        
          - ctc_input_len: input length of ctc (int)
          - batch_size:  (Int)
          - im_path: image path (String)
          - csv_path: training dataset path (Int)
          - cont: aux indice to extract in a seq way(Int)

        out:

          - batchx: Tensor with images as matrices for ANN input.
            (Array of Floats: [batch_size, height, width, 1])
          - sparse: SparseTensor with labels (SparseTensor: indice,values,shape)
          - transcriptions: Arraywith labels of "batchx". (Array de Strings: [batch_size])
          - num_samples: Number of samples (Int)
          
     """
    

    df = pd.read_csv(csv_path, sep=",",index_col="index")
    
    df_sample=df.loc[int(cont*batch_size):int((cont+1)*batch_size)-1,:].reset_index()
    
    num_samples=int(len(df_sample.axes[0]))
 




    batchx = []
    transcriptions = []
    index = []
    values=[]
    seq_len=[]

   
    if len(df_sample.axes[0]) is not 0:
        for i in range(len(df_sample.axes[0])):
            im_apt = df_sample.loc[i, ['image']].as_matrix()
     
            df_y =df_sample.loc[i, ['transcription']].as_matrix()
            for fich in im_apt:

       
                fich = str(fich)
                fich = fich.replace("['", "").replace("']", "")
                im = Image.open(im_path + fich + ".png")
                width, height = im.size
                im_map = list(im.getdata())
                im_map = np.array(im_map)
                im_map = im_map / 255
                result=im_map.reshape(height, width,1)
                batchx.append(result)

    
                original=""
                for n in list(str(df_y)):
                    if n == n.lower() and n == n.upper():
                        if n in "0123456789":
                            values.append(int(n))
                            original=original+n
                    elif n==n.lower():
                        values.append(int(ord(n)-61))
                        original = original + n
                    elif n==n.upper():
                        values.append(int((ord(n)-55)))
                        original = original + n

        
                for j in range(len(str(df_y))-4):
                    index.append([i,j])


                transcriptions.append(original)
                seq_len.append(ctc_input_len)

      
        batchx=np.stack(batchx, axis=0)
  
    shape=[batch_size,18]
    sparse=index,values,shape

    return batchx, sparse, transcriptions, seq_len, num_samples




def validation(curr_epoch,ctc_input_len, batch_size, im_path, csv_path, inputs, targets, keep_prob, seq_len, session, cost, ler):

    """
        
        Args:

          - curr_epoch: current epoch (Int)
          - ctc_input_len: input length of ctc (Int)
          - batch_size:  (Int)
          - im_path: images path (String)
          - csv_path: images dataset path (Int)
          - inputs: Placeholder of input (placeholder)
          - targets: Placeholder of output (placeholder)
          - keep_prob: Placeholder of dropout probability (placeholder)
          - seq_len: Placeholder for the length of the CTC input (placeholder)
          - session: TensorFlow session. (Session)
          - cost:CTC cost. (Tensor: [1])
          - ler: Tensor for LER. (Tensor:[1])

        out:

          - val_tuple: Result of the validation. (Tuple: {'epoch','cost','LER'})
    """

  
    cont = 0
    total_val_cost = 0
    total_val_ler = 0

    while cont >= 0:
  
        val_inputs, val_targets, val_original, val_seq_len, num_samples = extract_ordered_batch(
            ctc_input_len, batch_size, im_path, csv_path, cont)

     
        if num_samples == batch_size:
            val_feed = {inputs: val_inputs,
                    targets: val_targets,
                    keep_prob: 1,
                    seq_len: val_seq_len}
            val_cost, val_ler = session.run([cost, ler], val_feed)
            total_val_cost += val_cost
            total_val_ler += val_ler
            cont += 1

 
        elif num_samples == 0:
            val_tuple = {'epoch': [curr_epoch], 'val_cost': [total_val_cost / (cont + 1)],
                    'val_ler': [total_val_ler / (cont + 1)]}
            cont = -1

        else:
            val_feed = {inputs: val_inputs,
                    targets: val_targets,
                    keep_prob: 1,
                    seq_len: val_seq_len}
            val_cost, val_ler = session.run([cost, ler], val_feed)
            total_val_cost += val_cost
            total_val_ler += val_ler
            val_tuple = {'epoch': [curr_epoch], 'val_cost': [total_val_cost / (cont + 1)],
                    'val_ler': [total_val_ler / (cont + 1)]}
            cont = -1

    return val_tuple

