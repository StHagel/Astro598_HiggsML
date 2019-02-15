""" main.py
ASTR598 - Deep Learning final project by Stephan Hagel.

The purpose of this project is to take data from simulated proton-proton collisions in the ATLAS detector at the LHC in
Cern. This project was motivated by the 2014 ATLAS Higgs Machine Learning challenge. The data used for this project has
been provided for said challenge, which consists of 818238 events with 30 dimensions.
"""

import csv
import numpy as np


# START READING DATA
datafile = "../data/atlas-higgs-challenge-2014-v2.csv"
print("Reading data from " + datafile)
all_data = list(csv.reader(open(datafile,"r"), delimiter=','))

# This header will be used to identify, which field of the data corresponds to what label
header = all_data.pop(0)

# These next few lines extract some of the labels from the header for easy access.
iid = header.index("EventId")
immc = header.index("DER_mass_MMC")
ilabel = header.index("Label")
ikaggleset = header.index("KaggleSet")
ikaggleweight = header.index("KaggleWeight")
iweight = header.index("Weight")  # original weight
injet = header.index("PRI_jet_num")

# print(injet)
# print(iid)
# print(iweight)

# Since the data we read is still a string, we need to convert it to numbers. The EventID and number of jets are just
# Integers though, therefore they need to be handled separately.
for entry in all_data:
    for i in range(len(entry)):
        if i in [iid, injet]:
            entry[i] = int(entry[i])
        elif i not in [ilabel, ikaggleset]:
            entry[i] = float(entry[i])

# TODO: Convert list to numpy array

# END READING DATA

# START PREPARING DATA

# TODO: Handle -999.0 values
# TODO: Normalize if needed
# TODO: Split training- and testdata

# END PREPARING DATA

# START TRAINING MODEL

from keras.models import Sequential
from keras.layers import Dense, Dropout

# Here is where the actual model training begins. The parameters right now are taken from the example in the Keras
# documentation. Also the variable names for x_train and y_train have to be adjusted.
# TODO: Choose meaningful parameters, experiment with the model
model = Sequential()

model.add(Dense(64, input_dim=20, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

model.fit(x_train, y_train,
          epochs=20,
          batch_size=128)

score = model.evaluate(x_test, y_test, batch_size=128)

# END TRAINING MODEL

# START OPTIONAL CODE

# TODO: Implement some of the improvements mentioned in the HiggsML talk.

# END OPTIONAL CODE
