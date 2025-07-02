import os

from ultralytics import YOLO
from ultralytics.utils.torch_utils import de_parallel

from customs import CustomTrainer


os.environ["CUDA_VISIBLE_DEVICES"] = "3"

# # Функция для логирования метрик
# def log_custom_metrics(trainer):
#     print(trainer)
#     validator = de_parallel(trainer)
#     print(type(validator))
#     print(type(validator.metrics))
#     print(f"mAP@0.3: {validator.metrics.map30:.4f}")
    
#     if trainer.tb_writer:
#         trainer.tb_writer.add_scalar("metrics/mAP30", 
#                                     validator.metrics.map30, 
#                                     trainer.epoch)

if __name__ == "__main__":
    model = YOLO("my_yolo.yaml")
    # model.add_callback("on_val_end", log_custom_metrics)

    results = model.train(trainer=CustomTrainer, data="data.yaml", epochs=5,
                          batch=16, workers=1,
                          copy_paste=0.5, flipud=0.5,
                          perspective=0.00005, scale=0.3,
                          degrees=45, imgsz=320)

    # Результаты валидации
    # validation_results = model.val(validator=CustomValidator(model.args))
    print(f"Validation mAP@0.3: {results.metrics.map30:.4f}")
