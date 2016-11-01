# learn-gender-from-names

determine the gender from a given name with machine learning algorithms. The script uses the pybrain library to
create a neural network which tries to determine, if a given name is male or female. For this purpose the repository
contains a list of approx 10000 male and female names to learn

## Usage
clone this repository from git hub with
  
    git clone git@github.com:binspeak/learn-gender-from-names.git

make sure you have pybrain installed. If not use

    pip install pybrain

to get it.

Now you can directly run the script

    python learn-gender-from-names.py

this will train the neural network with the data given in data/ and output some statistics. 
You can also test explicit names by adding them to the command

   python learn-gender-by-names.py markus anika thorsten rebecca

In the output you will see the prediction for each name.

The repository also provides already trained weights for the network. If you want to use them instead of training, you can cange the switch use_stored_weights in the top section of the code to turn it on (set to True)
As long as this is set to False, each new training run will overwrite the weights (data/weights.json)
