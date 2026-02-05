# Copyright (c) Opendatalab. All rights reserved.
import io

from PIL import Image

from mineru.backend.pipeline.model_init import doclayout_yolo_model_init, mfd_model_init, mfr_model_init, ocr_model_init, wired_table_model_init, wireless_table_model_init, table_cls_model_init, layout_reader_model_init, PaddleOrientationClsModel, AtomModelSingleton
from mineru.backend.pipeline.model_json_to_middle_json import result_to_middle_json
from mineru.backend.pipeline.model_list import AtomicModel
from mineru.backend.pipeline.pipeline_analyze import ModelSingleton, batch_image_analyze
from mineru.cli.common import convert_pdf_bytes_to_bytes_by_pypdfium2
from mineru.data.data_reader_writer import DataWriter
from mineru.utils.pdf_classify import classify
from mineru.utils.pdf_image_tools import load_images_from_pdf


class InMemoryImageWriter(DataWriter):
    def __init__(self):
        self.files = {}

    def write(self, path: str, data: bytes) -> None:
        self.files[path] = Image.open(io.BytesIO(data))


def get_or_create_model(
    model_name,
    model_path=None,
    lang=None,
    device=None,
    det_db_box_thresh=0.3,
    det_db_unclip_ratio=1.8,
    enable_merge_det_boxes=True,
    ocr_model=None
):
    if model_name in [AtomicModel.WiredTable, AtomicModel.WirelessTable]:
        key = (
            model_name,
            lang
        )
    elif model_name in [AtomicModel.OCR]:
        key = (
            model_name,
            det_db_box_thresh,
            lang,
            det_db_unclip_ratio,
            enable_merge_det_boxes
        )
    else:
        key = model_name

    atom_model_manager = AtomModelSingleton()
    if key in atom_model_manager._models:
        return atom_model_manager._models[key]
    return set_model(model_name, model_path, lang, device, det_db_box_thresh, det_db_unclip_ratio, enable_merge_det_boxes, ocr_model)


def set_model(
    model_name,
    model_path=None,
    lang=None,
    device=None,
    det_db_box_thresh=0.3,
    det_db_unclip_ratio=1.8,
    enable_merge_det_boxes=True,
    ocr_model=None
):
    if model_name in [AtomicModel.WiredTable, AtomicModel.WirelessTable]:
        key = (
            model_name,
            lang
        )
    elif model_name in [AtomicModel.OCR]:
        key = (
            model_name,
            det_db_box_thresh,
            lang,
            det_db_unclip_ratio,
            enable_merge_det_boxes
        )
        print(key)
    else:
        key = model_name

    atom_model_manager = AtomModelSingleton()

    if model_name == AtomicModel.Layout:
        if model_path == None:
            raise Exception("Layout model needs weights")
        if device == None:
            raise Exception("Layout model needs device")
        atom_model = doclayout_yolo_model_init(model_path, device)
    elif model_name == AtomicModel.MFD:
        if model_path == None:
            raise Exception("MFD model needs weights")
        if device == None:
            raise Exception("Layout model needs device")
        atom_model = mfd_model_init(model_path, device)
    elif model_name == AtomicModel.MFR:
        if model_path == None:
            raise Exception("MFR model needs weights")
        if device == None:
            raise Exception("Layout model needs device")
        atom_model = mfr_model_init(model_path, device)
    elif model_name == AtomicModel.OCR:
        atom_model = ocr_model_init(
            model_path,
            det_db_box_thresh,
            lang,
            det_db_unclip_ratio,
            enable_merge_det_boxes)
    elif model_name == AtomicModel.WirelessTable:
        if ocr_model == None:
            raise Exception("You must provide an OCR model or a description")
        if type(ocr_model) == dict:
            ocr_model = get_or_create_model(**ocr_model)
        atom_model = wireless_table_model_init(model_path, ocr_model, lang)
    elif model_name == AtomicModel.WiredTable:
        if ocr_model == None:
            raise Exception("You must provide an OCR model or a description")
        if type(ocr_model) == dict:
            ocr_model = get_or_create_model(**ocr_model)
        atom_model = wired_table_model_init(model_path, ocr_model, lang)
    elif model_name == AtomicModel.TableCls:
        atom_model = table_cls_model_init(model_path)
    elif model_name == AtomicModel.ImgOrientationCls:
        if ocr_model == None:
            if ocr_model == None:
                raise Exception(
                    "You must provide an OCR model or a description")
            if type(ocr_model) == dict:
                ocr_model = get_or_create_model(**ocr_model)
        atom_model = PaddleOrientationClsModel(model_path, ocr_model)
    elif model_name == AtomicModel.LayoutReader:
        atom_model = layout_reader_model_init(model_path, device)
    else:
        raise Exception("Unknown model")

    if model_name in [AtomicModel.WiredTable, AtomicModel.WirelessTable]:
        key = (
            model_name,
            lang
        )
    elif model_name in [AtomicModel.OCR]:
        key = (
            model_name,
            det_db_box_thresh,
            lang,
            det_db_unclip_ratio,
            enable_merge_det_boxes
        )
    else:
        key = model_name

    atom_model_manager._models[key] = atom_model
    return atom_model


def initialise(model_descriptors):
    for descriptor in model_descriptors:
        set_model(**descriptor)


def infer(pdf_bytes, language='en', formula_enable=True, table_enable=True, batch_inference_size=384):
    model_manager = ModelSingleton()
    model_manager.get_model(
        lang=None,
        formula_enable=formula_enable,
        table_enable=table_enable,
    )

    data = convert_pdf_bytes_to_bytes_by_pypdfium2(pdf_bytes)

    use_ocr = False
    if classify(data) == 'ocr':
        use_ocr = True

    images_list, pdf_doc = load_images_from_pdf(data, image_type='pil_img')

    all_pages_info = []
    for page_img in images_list:
        all_pages_info.append((page_img['img_pil'], use_ocr, language))

    batch_size = batch_inference_size
    batch_images = [
        all_pages_info[i:i + batch_size]
        for i in range(0, len(all_pages_info), batch_size)
    ]
    results = []
    for batch_image in batch_images:
        batch_results = batch_image_analyze(
            batch_image, formula_enable, table_enable)
        results.extend(batch_results)

    infer_results = []

    for i, page_info in enumerate(all_pages_info):
        pil_img, _, _ = page_info
        result = results[i]

        page_info_dict = {'page_no': i,
                          'width': pil_img.width, 'height': pil_img.height}
        page_dict = {'layout_dets': result, 'page_info': page_info_dict}

        infer_results.append(page_dict)
    img_writer = InMemoryImageWriter()
    json_data = result_to_middle_json(
        infer_results, images_list, pdf_doc, img_writer, language, use_ocr, formula_enable)
    return json_data, img_writer.files
