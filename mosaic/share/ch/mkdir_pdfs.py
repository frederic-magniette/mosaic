from glob import glob
<<<<<<< HEAD
from shutil import copyfile
=======
from shutil import move as mv
>>>>>>> 9a111c7 (removing useless outputs)

def move():
    pdfs = glob('*.pdf')
    for pdf in pdfs:
<<<<<<< HEAD
        copyfile(pdf, 'pdfs/'+pdf)
=======
        mv(pdf, 'pdfs/'+pdf)
>>>>>>> 9a111c7 (removing useless outputs)
