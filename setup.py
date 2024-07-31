from setuptools import setup, find_packages

install_requires=[]
try :
    with open("requirements.txt",'r') as fp:
        install_requires = fp.read()
except OSError:
    print('Error encountered when opening requirements file')
    
setup(
	name="PAM_Diago",
	version="1.0.1",
	packages=find_packages(exclude=['run','pcc']), # permet de récupérer tout les fichiers 
	description="PAM metric for audio quality assessement",
	author="Diago",
	license="WTFPL",
	python_requires=">=3.6",
    install_requires=install_requires
	)