""" main.py
ASTR598 - Deep Learning final project by Stephan Hagel.

The purpose of this project is to take data from simulated proton-proton collisions in the ATLAS detector at the LHC in
Cern. This project was motivated by the 2014 ATLAS Higgs Machine Learning challenge. The data used for this project has
been provided for said challenge, which consists of 818238 events with 30 dimensions.

Some notes about the dataframe:
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

USAGE: python3 main.py flags
where flags can be one or more of
IGNORE_MASS_DATA
IGNORE_JET_DATA
REMOVE_HIGGS_NAN
SIMPLE_IMPUTE
ADVANCED_IMPUTE
"""

import sys
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from keras import backend as K
import tensorflow as tf

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
IGNORE_MASS_DATA = False
IGNORE_JET_DATA = False
REMOVE_HIGGS_NAN = False
SIMPLE_IMPUTE = False
ADVANCED_IMPUTE = False

PHYSICAL_HIGGS_MASS = 125.18


def main():
    # START GPU CONFIG

    config = tf.ConfigProto( device_count = {'GPU': 1 , 'CPU': 24} ) 
    sess = tf.Session(config=config) 
    keras.backend.set_session(sess)
	
	# END GPU CONFIG
    # START SETTING FLAGS

    global IGNORE_MASS_DATA, IGNORE_JET_DATA, REMOVE_HIGGS_NAN, SIMPLE_IMPUTE, ADVANCED_IMPUTE

    if "IGNORE_JET_DATA" in sys.argv:
        print("Ignoring jet data.")
        IGNORE_JET_DATA = True

    if "IGNORE_MASS_DATA" in sys.argv:
        print("Ignoring mass data.")
        IGNORE_MASS_DATA = True

    if "REMOVE_HIGGS_NAN" in sys.argv:
        print("Removing NaN's in Higgs mass.")
        REMOVE_HIGGS_NAN = True

    if "SIMPLE_IMPUTE" in sys.argv:
        print("Using simple imputer for Higgs mass.")
        SIMPLE_IMPUTE = True

    if "ADVANCED_IMPUTE" in sys.argv:
        print("Using advanced imputer for Higgs massE")
        ADVANCED_IMPUTE = True

    # END SETTING FLAGS

    # START READING DATA
    try:
        datafile = "../data/data.csv"
        print("Reading data from " + datafile)
        dataframe = pd.read_csv(datafile, header=None)
    except FileNotFoundError:
        print("File " + datafile + " not found. Please make sure you are running this program from the src folder.")
        quit()

    # First we delete the columns that contain data, which will not be used to in the model
    print("Removing unused columns.")
    del dataframe[KAGGLE_WEIGHT_INDEX]
    del dataframe[KAGGLE_SET_INDEX]
    del dataframe[WEIGHT_INDEX]
    del dataframe[EVENTID_INDEX]

    # And we also remove the header
    dataframe = dataframe.iloc[1:]

    # Next we convert the strings in the dataframe to actual numbers. The option errors='ignore' guarantees, that the
    # values for the label are kept and not converted to NaN.
    print("Converting to numbers.")
    dataframe = dataframe.apply(pd.to_numeric, errors='ignore')

    # Now we can replace the -999.0 entries with NaN.
    print("Replacing missing data with NaN.")
    dataframe = dataframe.replace({-999.0: np.NaN})

    # We will further convert the label from 'b' and 's' to 0 and 1 respectively.
    print("Converting labels.")
    dataframe[LABEL_INDEX] = 1 - pd.factorize(dataframe[LABEL_INDEX])[0]

    # END READING DATA

    # START PREPARING DATA

    # Depending on the option set above, we will handle the NaN's in different ways:
    if IGNORE_JET_DATA:
        # Case 1: Just ignore the columns with jet dara in them.
        print("Mode IGNORE_JET_DATA is set to True.")
        print("Deleting missing data.")
        for i in SOLOJET_INDEX:
            del dataframe[i]
        for j in MULTIJET_INDEX:
            del dataframe[j]

        # If the IGNORE_MASS_DATA flag is set, we will also delete the mass column.
        if IGNORE_MASS_DATA:
            del dataframe[DER_MASS_INDEX]
        # If the REMOVE_HIGGS_NAN flag is set, we remove the NaN's.
        elif REMOVE_HIGGS_NAN:
            dataframe.dropna(inplace=True)
        # For the simple imputing we use the physical higgs mass to replace NaN's.
        elif SIMPLE_IMPUTE:
            dataframe.fillna(PHYSICAL_HIGGS_MASS, inplace=True)

    # TODO 2: Remove the lines with NaN's in the Higgs mass and split the data by the number of jets
    # TODO 3: Use physical value (or mean?) for the Higgs mass as imputation.
    # TODO 4: Use a regression model as Higgs mass imputer (advanced)

    # Now we can convert the daraframe to a numpy matrix.
    print("Converting data to Matrix.")
    data_matrix = dataframe.as_matrix().astype(np.float)

    # Let's do some garbage collection.
    del dataframe

    # We will use the MinMaxScaler form sklearn to normalize the data
    scaler = MinMaxScaler()
    scaler.fit(data_matrix)
    print("Normalizing data.")
    data_matrix_norm = scaler.transform(data_matrix)
    del data_matrix

    # We furthermore need to separate the labels from the actual training data
    print("Separating labels.")
    target = data_matrix_norm[:, -1]
    train = data_matrix_norm[:, :-1]
    del data_matrix_norm

    # Now we can finally split our data into training and test data and start training our model
    print("Splitting test and training data")
    x_train, x_test, y_train, y_test = train_test_split(train, target, test_size=0.15, random_state=1)

    print("Finished data preparation.")
    # END PREPARING DATA

    # START TRAINING MODEL

    # Since the code for training the model with the data from jet events differs considerably from
    if IGNORE_JET_DATA:
        print(train_nojet(x_train, x_test, y_train, y_test, len(train[0])))

    # END TRAINING MODEL

    # START OPTIONAL CODE

    # TODO: Implement some of the improvements mentioned in the HiggsML talk.

    # END OPTIONAL CODE


def train_nojet(x_train, x_test, y_train, y_test, input_dim_):
    from keras.models import Sequential
    from keras.layers import Dense, Dropout

    # Here is where the actual model training begins. The parameters right now are taken from the example in the Keras
    # documentation. Also the variable names for x_train and y_train have to be adjusted.
    # TODO: Choose meaningful parameters, experiment with the model
    model = Sequential()

    model.add(Dense(input_dim_, input_dim=input_dim_, activation='relu'))
    # model.add(Dense(64, activation='relu'))
    # model.add(Dropout(0.25))
    # model.add(Dense(128, activation='relu'))
    # model.add(Dropout(0.5))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])

    model.fit(x_train, y_train,
              epochs=20,
              batch_size=128)

    return model.evaluate(x_test, y_test, batch_size=128)


if __name__ == "__main__":
    main()
