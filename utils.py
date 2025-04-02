# файлик с моими стандартыми утилитами
import torch
from numba import cuda

def check_gpu(logger=None):
    """
    Проверка доступности GPU
    """
    try:
        if cuda.is_available() and torch.cuda.is_available():
            if logger:
                logger.info(f"Доступно GPU: {len(cuda.gpus)}")
                logger.info(f"Название GPU: {cuda.gpus[0].name.decode()}")
            else:
                print('Доступно GPU:', len(cuda.gpus))
                print('Название GPU:', cuda.gpus[0].name.decode())
            return
        
        if logger:
            logger.error("GPU не обнаружены!")
        else:
            print("GPU не обнаружены!")
        return False
    
    except Exception as e:
        if logger:
            logger.error(f"Ошибка при проверке GPU: {e}")
        else:
            print(f"Ошибка при проверке GPU: {e}")
        return False

