""" main.py
ASTR598 - Deep Learning final project by Stephan Hagel.

The purpose of this project is to take data from simulated proton-proton collisions in the ATLAS detector at the LHC in
Cern. This project was motivated by the 2014 ATLAS Higgs Machine Learning challenge. The data used for this project has
been provided for said challenge, which consists of 818238 events with 30 dimensions.

Some notes about the dataset:
The following indices are of significant importance:

index 0: EventID
    An integer, that uniquely labels each event. Not to be used as an input parameter.

index 1: DER_mass_MMC
    The mass of the Higgs candidate. Should be around the physical value of ~125 GeV.
    Contains 122143 points of missing data, which have to be taken care of.

index 5-7:
    Only defined, if two or more jets are generated. 568698 points of systematically missing data.

index 13:
    Only defined, if two or more jets are generated. 568698 points of systematically missing data.

index 23: PRI_jet_num
    An integer that indicates, how many jets have been generated in that event.

index 24-26:
    Only defined, if one or more jets are generated. 320850 points of systematically missing data.

index 27-29:
    Only defined, if two or more jets are generated. 568698 points of systematically missing data.

index 31: Weight
    Used in the original challenge, obsolete for this project.

index 32: Label
    The actual label that is used for the classification. 'b' stands for background, 's' stands for signal.

index 33-34: Kaggle*
    Used in the original challenge, obsolete for this project.
"""

import numpy as np
import pandas as pd
import csv

# These are the important indices mentioned above.
KAGGLE_WEIGHT_INDEX = 34
KAGGLE_SET_INDEX = 33
WEIGHT_INDEX = 31
LABEL_INDEX = 32
EVENTID_INDEX = 0
DER_MASS_INDEX = 1
MULTIJET_INDEX = [5, 6, 7, 13, 27, 28, 29]
JETNUMBER_INDEX = 23
SOLOJET_INDEX = [24, 25, 26]

# These Booleans will define how the missing data is handled.
IGNORE_MISSING_DATA = True
REMOVE_HIGGS_NAN = False
SIMPLE_IMPUTE = False
ADVANCED_IMPUTE = False


def main():
    # START READING DATA
    datafile = "../data/data.csv"
    print("Reading data from " + datafile)
    dataset = pd.read_csv(datafile, header=None)

    # First we delete the coloumns that contain data, which will not be used to in the model
    del dataset[KAGGLE_WEIGHT_INDEX]
    del dataset[KAGGLE_SET_INDEX]
    del dataset[WEIGHT_INDEX]
    del dataset[EVENTID_INDEX]

    # And we also remove the header
    dataset = dataset.iloc[1:]

    # Next we convert the strings in the dataset to actual numbers. The option errors='ignore' guarantees, that the
    # values for the label are kept and not converted to NaN.
    dataset = dataset.apply(pd.to_numeric, errors='ignore')

    dataset = dataset.replace({-999.0: np.NaN})

    # TODO: Convert the 's' and 'b' labels to 1 and 0.

    # END READING DATA

    # START PREPARING DATA

    # Depending on the option set above, we will handle the NaN's in different ways:
    if IGNORE_MISSING_DATA:
        # Case 1: Just ignore the columns with NaN's in them
        del dataset[DER_MASS_INDEX]
        for i in SOLOJET_INDEX:
            del dataset[i]
        for j in MULTIJET_INDEX:
            del dataset[j]

    print("Works so far")

    # TODO 2: Remove the lines with NaN's in the Higgs mass and split the data by the number of jets
    # TODO 3: Use physical value (or mean?) for the Higgs mass as imputation.
    # TODO 4: Use a regression model as Higgs mass imputer (advanced)

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


if __name__ == "__main__":
    main()
