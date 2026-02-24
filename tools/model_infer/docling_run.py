from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.datamodel.base_models import InputFormat
import os
from tqdm import tqdm


pipeline_options = PdfPipelineOptions()
pipeline_options.do_formula_enrichment = True  # Added
pipeline_options.do_ocr = True


converter = DocumentConverter(
    format_options={
        InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
    }
)

img_path = './pdfs'
save_path = './result/docling'


for img_name in tqdm(os.listdir(img_path)):
    if not img_name.endswith('.pdf'):
        continue
    
    img_name = img_name.strip()
    save_result_path = os.path.join(save_path, img_name[:-4] + '.md')
    
    if os.path.exists(save_result_path):
        continue

    img_path_tmp = os.path.join(img_path, img_name)
    try:
        result = converter.convert(img_path_tmp)
        result_md = result.document.export_to_markdown()
    except Exception as e:
        print(f"Failed: {img_name} - {e}")
        continue

    with open(save_result_path, 'w', encoding='utf-8') as output_file:
        output_file.write(result_md)