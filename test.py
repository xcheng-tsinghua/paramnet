
import torch
import os
import numpy as np
from sklearn.preprocessing import label_binarize, LabelBinarizer


def test():
    # aTensor = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 1, 4])
    # bTensor = torch.tensor([1, 3, 2, 2, 5, 5, 2, 5, 1, 3])
    #
    # cro = (aTensor == 1) & (bTensor == 1)
    # uni = (aTensor == 1) | (bTensor == 1)
    #
    # print(cro)
    # print(aTensor[cro])
    # print(uni)
    filepath_all = []

    n_file_count = 0
    for root, dirs, files in os.walk(r'D:\document\DeepLearning\DataSet\STEP9000\raw'):
        for file in files:

            n_file_count += 1

    print(n_file_count)


if __name__ == '__main__':
    # test()
    cla = 2
    alabel = torch.randint(0, cla, (10,)).numpy()

    print(alabel)
    bainsas = label_binarize(alabel, classes=np.arange(cla))

    if cla == 2:
        bainsas_rev = 1 - bainsas
        bainsas = np.concatenate([bainsas_rev, bainsas], axis=1)

    print(bainsas)


    pass

