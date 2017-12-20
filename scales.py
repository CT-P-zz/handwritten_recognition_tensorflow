def scale_invert():
    raw_path="C:/Users/catarina/QRCodeDetection/Offline-Handwriting-Recognition-with-TensorFlow/Data/Dataset_words"
    proc_path="C:/Users/catarina/QRCodeDetection/Offline-Handwriting-Recognition-with-TensorFlow/Data/Dataset_words_resized"
    height=1024
    width=64
    print('here')
    """
    Función que escala e invierte cada imagen para almacenarlas en un directorio común.
    Se conservan las proporciones de la imagen original y se añade un relleno hasta alcanzar
    el ancho objetivo.

    Argumentos:

      - raw_path: Ruta de la imagen original. (String)
      - proc_path: Ruta donde almacenar la imagen procesada. (String)
      - height: Altura de las imágenes. (Int)
      - width: Anchura de la imágenes. (Int)

    """
    # Cargamos la imagen
    im = Image.open(raw_path)
    print('open image')
    # Reescalamos
    raw_width, raw_height = im.size
    new_width = int(round(raw_width * (height / raw_height)))
    im = im.resize((new_width, height), Image.NEAREST)
    im_map = list(im.getdata())
    im_map = np.array(im_map)
    im_map = im_map.reshape(height, new_width).astype(np.uint8)

    # Rellenamos e invertimos los valores.
    data = np.full((height, width - new_width + 1), 255)
    im_map = np.concatenate((im_map, data), axis=1)
    im_map = im_map[:, 0:width]
    im_map = (255 - im_map)
    im_map = im_map.astype(np.uint8)
    im = Image.fromarray(im_map)

    # Almacenamos todas las imágenes en directorio común
    im.save(str(proc_path), ".png")
    print("Processed image saved: " + str(proc_path))
    
