from pypdf import PdfReader


# Function to read PDF files
def read_pdf(file_path):
    reader = PdfReader(open(file_path, "rb"))
    text = ""
    for page_num in range(reader.get_num_pages()):
        text += reader.get_page(page_num).extract_text()
    return text
