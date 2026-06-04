from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import as_completed
from utils import *
import os

def process_images():
    input_dir = Path('data/raw/test')
    output_dir = Path('data/processed/test')

    files = [f for f in os.listdir(input_dir) if f.endswith('.dicom')]
    
    tasks = [
        (input_dir / f, output_dir / f.replace('.dicom', '.png'), 1024) 
        for f in files
    ]

    print(f"Найдено файлов: {len(tasks)}")
    print(f"Запуск параллельной обработки на {os.cpu_count()} ядрах...")


    with ProcessPoolExecutor() as executor:
        futures = []
        for f in files:
            obj = executor.submit(
                process_image, 
                input_path=input_dir / f, 
                output_path=output_dir / f.replace('.dicom', '.png'), 
                size=1024
            )
            futures.append(obj)

        
        for _ in tqdm(as_completed(futures), total=len(futures)):
            pass

    print("Обработка test завершена.")




    input_dir = Path('data/raw/train')
    output_dir = Path('data/processed/train')

    files = [f for f in os.listdir(input_dir) if f.endswith('.dicom')]
    
    tasks = [
        (input_dir / f, output_dir / f.replace('.dicom', '.png'), 1024) 
        for f in files
    ]

    print(f"Найдено файлов: {len(tasks)}")
    print(f"Запуск параллельной обработки на {os.cpu_count()} ядрах...")


    with ProcessPoolExecutor() as executor:
        futures = []
        for f in files:
            obj = executor.submit(
                process_image, 
                input_path=input_dir / f, 
                output_path=output_dir / f.replace('.dicom', '.png'), 
                size=1024
            )
            futures.append(obj)

        
        for _ in tqdm(as_completed(futures), total=len(futures)):
            pass

    print("Обработка train завершена.")


if __name__ == '__main__':
    process_images()