from glob import glob
from shutil import copyfile

def move():
    pdfs = glob('*.pdf')
    for pdf in pdfs:
        copyfile(pdf, 'pdfs/'+pdf)
