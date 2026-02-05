import mineru
from mineru import AtomicModel
import os

if __name__ == "__main__":

    mineru.initialise([
        {
            "model_name": AtomicModel.Layout,
            "model_path": os.path.join(os.getcwd(), "models/Layout/YOLO/doclayout_yolo_docstructbench_imgsz1280_2501.pt"),
            "device": "cuda"
        },
        {
            "model_name": AtomicModel.MFD,
            "model_path": os.path.join(os.getcwd(), "models/MFD/YOLO/yolo_v8_ft.pt"),
            "device": "cuda"
        },
        {
            "model_name": AtomicModel.MFR,
            "model_path": os.path.join(os.getcwd(), "models/MFR/unimernet_hf_small_2503"),
            "device": "cuda"
        },
        {
            "model_name": AtomicModel.OCR,
            "model_path": os.path.join(os.getcwd(), "models/OCR/paddleocr_torch"),
            "device": "cuda",
            "lang": "en"
        },
        {
            "model_name": AtomicModel.WirelessTable,
            "model_path": os.path.join(os.getcwd(), "models/TabRec/SlanetPlus/slanet-plus.onnx"),
            "ocr_model": {
                "model_name": AtomicModel.OCR,
                "model_path": os.path.join(os.getcwd(), "models/OCR/paddleocr_torch"),
                "device": "cuda",
                "det_db_box_thresh": 0.3,
                "det_db_unclip_ratio": 1.8,
                "enable_merge_det_boxes": True
            }
        },
        {
            "model_name": AtomicModel.WiredTable,
            "model_path": os.path.join(os.getcwd(), "models/TabRec/UnetStructure/unet.onnx"),
            "ocr_model": {
                "model_name": AtomicModel.OCR,
                "model_path": os.path.join(os.getcwd(), "models/OCR/paddleocr_torch"),
                "device": "cuda",
                "det_db_box_thresh": 0.5,
                "det_db_unclip_ratio": 1.6,
                "enable_merge_det_boxes": False
            }
        },
        {
            "model_name": AtomicModel.TableCls,
            "model_path": os.path.join(os.getcwd(), "models/TabCls/paddle_table_cls/PP-LCNet_x1_0_table_cls.onnx"),
        },
        {
            "model_name": AtomicModel.ImgOrientationCls,
            "model_path": os.path.join(os.getcwd(), "models/TabCls/paddle_table_cls/PP-LCNet_x1_0_table_cls.onnx"),
            "ocr_model": {
                "model_name": AtomicModel.OCR,
                "model_path": os.path.join(os.getcwd(), "models/OCR/paddleocr_torch"),
                "device": "cuda",
                "det_db_box_thresh": 0.5,
                "det_db_unclip_ratio": 1.6,
                "enable_merge_det_boxes": False,
                "lang": "ch_lite",
            }
        },
        {
            "model_name": AtomicModel.LayoutReader,
            "model_path": os.path.join(os.getcwd(), "models/ReadingOrder/layout_reader"),
            "device": "cuda"
        }

    ])

    with open("/home/szabi/work/emission/pdf/output.pdf", "rb") as f:
        data = f.read()
    json_document, images = mineru.infer(data)
