import os
import cv2

import matplotlib.pyplot as plt
import numpy as np

from typing import List, Tuple
# from PIL import Image

def chargementImagesTrainTest(pathDir:str) -> Tuple[list]:
    # Ouverture du répertoire
    print(f"Ouverture du dossier : {pathDir}")
    print("---------------------------------")

    # Lecture des fichiers du répertoire
    dirsData = os.listdir(pathDir)
    desktopIni = "desktop.ini"

    # Ignorer fichier créés par Drive
    try :
        dirsData.pop(dirsData.index("desktop.ini"))
    except ValueError :
        pass

    #Initialisation des datasets
    #TODO: Checker la première valeur de ces listes
    X_train = np.empty([0, 28, 28, 3])
    y_train = np.empty([0])
    X_test = np.empty([0, 28, 28, 3])
    y_test = np.empty([0])

    # Parcours des sous-répertoires
    for dirData in dirsData:

        # Ouverture des sous-répertoires
        print(f"Ouverture du dossier : {pathDir}/{dirData}")
        print("------------------------------------------------------")

        # Lecture des fichiers du sous-répertoire
        dirsDigit = os.listdir(f"{pathDir}/{dirData}")

        # Ignorer fichier créés par Drive
        if desktopIni in dirsDigit:
            dirsDigit.pop(dirsDigit.index("desktop.ini"))
        
        # Test des noms de sous-répertoires
        #TODO: factoriser le code exécuté à l'aide d'une fonction
        if "training" in dirData:   # Train

            # Parcours des sous-sous-répertoires
            for dirDigit in dirsDigit:

                #Ouverture des sous-sous-répertoires(images)
                print(f"Ouverture du dossier : {pathDir}/{dirData}/{dirDigit}")
                print("------------------------------------------------------")

                # Parcours des sous-sous-répertoires(images)
                filesImages = os.listdir(f"{pathDir}/{dirData}/{dirDigit}")

                # Ignorer fichier créés par Drive
                if desktopIni in filesImages:
                    filesImages.pop(filesImages.index("desktop.ini"))

                # Affectation des datasets
                X_train = np.append(X_train, [cv2.imread(f"{pathDir}/{dirData}/{dirDigit}/{fileImage}") for fileImage in filesImages], axis=0)
                # X_train.append([cv2.imread(f"{pathDir}/{dirData}/{dirDigit}/{fileImage}") for fileImage in filesImages])
                y_train = np.append(y_train, [dirDigit for fileImage in filesImages], axis=0)
                # y_train.append([dirDigit for fileImage in filesImages])
            
        elif "test" in dirData: # Test

            # Parcours des sous-sous-répertoires
            for dirDigit in dirsDigit:      

                #Ouverture des sous-sous-répertoires(images)          
                print(f"Ouverture du dossier : {pathDir}/{dirData}/{dirDigit}")
                print("------------------------------------------------------")

                # Parcours des sous-sous-répertoires(images)
                filesImages = os.listdir(f"{pathDir}/{dirData}/{dirDigit}")

                # Ignorer fichier créés par Drive
                if desktopIni in filesImages:
                    filesImages.pop(filesImages.index("desktop.ini"))

                # Affectation des datasets
                X_test = np.append(X_test, [cv2.imread(f"{pathDir}/{dirData}/{dirDigit}/{fileImage}") for fileImage in filesImages], axis=0)
                y_test = np.append(y_test, [dirDigit for fileImage in filesImages], axis=0)
                # X_test.append([cv2.imread(f"{pathDir}/{dirData}/{dirDigit}/{fileImage}") for fileImage in filesImages])
                # y_test.append([dirDigit for fileImage in filesImages])

        else:
            pass

    # Normalisation
    X_train, X_test = X_train/255.0, X_test/255.0


    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = chargementImagesTrainTest("data")

    plt.imshow(X_train[500])
    plt.show()
    print(y_train[500])