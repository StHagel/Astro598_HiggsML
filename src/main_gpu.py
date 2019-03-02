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
IGNORE_MULTIJET_DATA
REMOVE_HIGGS_NAN
SIMPLE_IMPUTE
"""

import sys
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from keras import backend as K
from keras import save_model_hdf5
from keras import load_model_hdf5
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
IGNORE_MULTIJET_DATA = False
REMOVE_HIGGS_NAN = False
SIMPLE_IMPUTE = False

# This Boolean shows, if the model should be saved in the end.
SAVE_MODEL = False

# This Boolean shows, if a model should be loaded from a file. If this is the case, no new model will be trained.
LOAD_MODEL = False

PHYSICAL_HIGGS_MASS = 125.18


def main():
    # START GPU CONFIG

    config = tf.ConfigProto(device_count={'GPU': 1, 'CPU': 24})
    sess = tf.Session(config=config)
    keras.backend.set_session(sess)

    # END GPU CONFIG
    # START SETTING FLAGS

    global IGNORE_MASS_DATA, IGNORE_JET_DATA, REMOVE_HIGGS_NAN, SIMPLE_IMPUTE, ADVANCED_IMPUTE, IGNORE_MULTIJET_DATA

    if "IGNORE_JET_DATA" in sys.argv:
        print("Ignoring all jet data.")
        IGNORE_JET_DATA = True

    if "IGNORE_MULTIJET_DATA" in sys.argv:
        print("Ignoring data with multiple jets.")
        IGNORE_MULTIJET_DATA = True

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
        print("Using advanced imputer for Higgs mass.")
        ADVANCED_IMPUTE = True

    # Set the flags for saving and loading the model.
    global SAVE_MODEL, LOAD_MODEL
    model_path = "models/model.h5"

    if "LOAD_MODEL" in sys.argv:
        LOAD_MODEL = True
        print("Model will be loaded from " + model_path + ".")

    if "SAVE_MODEL" in sys.argv:
        SAVE_MODEL = True
        print("Model will be saved to " + model_path + ".")

    # END SETTING FLAGS

    # START READING DATA
    try:
        datafile = "data/data.csv"
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

    elif IGNORE_MULTIJET_DATA:
        # Case 2: Ignore the data with multiple jets generated.
        print("Mode IGNORE_MULTIJET_DATA is set to True.")
        print("Deleting missing data.")

        # Delete the columns, that contain Multijet data
        for i in MULTIJET_INDEX:
            del dataframe[i]

        # Deleting the rows, which contain no jet data
        dataframe = dataframe[dataframe[JETNUMBER_INDEX] > 0]

    else:
        # Case 3: Using all jet data
        print("Only Events with multiple jets produced will be used.")
        print("Deleting missing data.")

        # Delete the rows with no jets produced
        dataframe = dataframe[dataframe[JETNUMBER_INDEX] > 1]

    # If the IGNORE_MASS_DATA flag is set, we will also delete the mass column.
    if IGNORE_MASS_DATA:
        del dataframe[DER_MASS_INDEX]
    # If the REMOVE_HIGGS_NAN flag is set, we remove the NaN's.
    elif REMOVE_HIGGS_NAN:
        dataframe.dropna(inplace=True)
    # For the simple imputing we use the physical higgs mass to replace NaN's.
    elif SIMPLE_IMPUTE:
        dataframe.fillna(PHYSICAL_HIGGS_MASS, inplace=True)

    # Now we can convert the dataframe to a numpy matrix.
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

    # Depending on the flags, that have been set, the dimension of our training data might vary.
    # Therefore we need to extract the input dimension to use it to make our network a reasonable size.
    input_dim_ = len(train[0])

    # Now we can finally split our data into training and test data and start training our model
    print("Splitting test and training data")
    x_train, x_test, y_train, y_test = train_test_split(train, target, test_size=0.15, random_state=1)

    print("Finished data preparation.")
    # END PREPARING DATA

    # START TRAINING MODEL

    from keras.models import Sequential
    from keras.layers import Dense, Dropout

    if not LOAD_MODEL:
        # Here is where the actual model training begins.
        model = Sequential()

        mean = (input_dim_ + 1) // 2

        model.add(Dense(input_dim_, input_dim=input_dim_, activation='relu'))
        model.add(Dense(mean, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))

        model.compile(loss='binary_crossentropy',
                      optimizer='rmsprop',
                      metrics=['accuracy'])

    else:
        model = load_model_hdf5(model_path)

    model.fit(x_train, y_train,
              epochs=20,
              batch_size=128)

    loss_and_metrics = model.evaluate(x_test, y_test, batch_size=128)
    print(loss_and_metrics)

    if SAVE_MODEL:
        save_model_hdf5(model, model_path)

    # END TRAINING MODEL

    # START OPTIONAL CODE

    # TODO: Implement some of the improvements mentioned in the HiggsML talk.

    # END OPTIONAL CODE


if __name__ == "__main__":
    main()
