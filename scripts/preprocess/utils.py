import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import cv2

def add_padding(img: np.ndarray) -> np.ndarray:
    """Делает изображение квадратным, добавляя черные полосы по краям.
    
    Метод сохраняет оригинальные пропорции объекта, предотвращая искажения
    анатомических структур при последующем изменении размера.

    Args:
        img: Входной массив изображения (NumPy array).

    Returns:
        np.ndarray: Квадратное изображение с симметричными отступами (padding).
    """

    h, w = img.shape[:2]
    
    side = max(h, w)
    
    top = (side - h) // 2
    bottom = side - h - top
    left = (side - w) // 2
    right = side - w - left
    
    return cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)


def process_image(input_path: Path, output_path: Path, size: int = 1024):
    """Выполняет обработку DICOM-файла и сохраняет его в формате PNG.
   
    Подробное описание логики:
    1. Приведение яркости к диапазону (0-255).
    2. Применение CLAHE.
    3. Приведение изображения к квадратным размерам.

    Args:
        input_path: Путь к исходному файлу формата .dcm.
        output_path: Путь к выходному файлу формата .png
        size: Размер стороны квадрата выходного изображения. По умолчанию 1024.

    Returns:
        None. Функция сохраняет файл на диск.
    """

    try:

        ds = pydicom.dcmread(input_path)

        img = apply_voi_lut(ds.pixel_array, ds)

        if ds.PhotometricInterpretation == "MONOCHROME1":
            img = np.amax(img) - img

        img = img - np.min(img)
        img = img / np.max(img)
        img = (img * 255).astype(np.uint8)

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img = clahe.apply(img)

        img = add_padding(img)
        img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)

        cv2.imwrite(output_path, img)

    except Exception as e:
        with open("processing_errors.log", "a") as f:
            f.write(f"Error processing {input_path}: {e}\n")



def process_bounding_box(origin_coords: list[float], origin_size: list[int], new_size: int):
    """Пересчитывает координаты Bounding Box с учетом добавленного паддинга и ресайза.
    
    ВНИМАНИЕ: Данная функция учитывает, что перед ресайзом изображение было 
    приведено к квадрату через `add_padding`.

    Args:
        origin_coords: Список исходных координат [x_min, y_min, x_max, y_max].
        origin_size: Исходный размер изображения (width_old, height_old).
        new_size: Новый размер квадратного изображения после обработки side_new.

    Returns:
        tuple: Новые координаты в формате (x_min_new, y_min_new, x_max_new, y_max_new).
    """

    width_old, height_old = origin_size
    side_new = new_size
    x_min, y_min, x_max, y_max = origin_coords

    side_old = max(width_old, height_old)
    offset_x = (side_old - width_old) // 2
    offset_y = (side_old - height_old) // 2

    scale = side_new / side_old

    x_min_new = (x_min + offset_x) * scale
    y_min_new = (y_min + offset_y) * scale
    x_max_new = (x_max + offset_x) * scale
    y_max_new = (y_max + offset_y) * scale

    return x_min_new, y_min_new, x_max_new, y_max_new