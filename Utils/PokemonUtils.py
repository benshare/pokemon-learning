import csv
import numpy as np
import matplotlib.pyplot as plt
import os.path  # File path checking
from PIL import Image, ImageSequence  # Reading GIFs


"""
PokemonUtils
------------
* Useage: utils = PokemonUtils()


DATA:
-----
* utils.x_train, .x_val, .x_test
(N x 1) arrays of pokedex numbers

* .y_train, .y_val, .y_test
(N x 1) arrays of type numbers (1 = Normal, ..., 18 = Fairy)


GIF MANIPULATION:
-----------------
* utils.readGif(path)
gif path name --> numpy array (T x H x W x C)


LOOKUP TABLES:
--------------
* utils.typeToNameDict, .nameToTypeDict
* utils.typeToName(n)
Dictionaries of type number <-> type name

* utils.nameToNumberDict, .NumberToNameDict
* utils.nameToNumber(name), .numberToName(n)
Dictionaries of pokedex number <-> pokemon name


OTHER:
------
* utils.uniqueValidIDs
List of pokedex numbers of all pokemon we will be using

* utils.uniqueValidPokemonNames
List of names of all pokemon we will be using

"""

class PokemonUtils():

    def __init__(self, verbose=True):
        self.csv_path = "../Data/veekun/"
        self.gifs_path = "../Data/pkparaiso/"
        self.splits_path = "../Data/Splits/"
        self.verbose = verbose
        self.generateNameToNumber()  # loads all names/numbers from veekun file
        self.generateMissingImages(self.gifs_path)  # cross-references with downloaded images
        self.generateNumberToTypes()  # Only for pokemon that have associated images
        self.generateTypeNames()  # Only for pokemon that have associated images
        self.generateTrainValTestSplit()  # Load from file

    def generateNameToNumber(self):
        self.nameToNumberDict = {}
        self.numberToNameDict = {}
        self.uniqueIDs = []
        with open(self.csv_path + "pokemon.csv") as f:
            reader = csv.DictReader(f)
            for row in reader:
                id = int(row['id'])
                name = self.normalizeString(row['identifier'])
                self.uniqueIDs.append(id)
                self.nameToNumberDict[name] = id
                self.numberToNameDict[id] = name

        self.allNames = self.numbersToName(self.uniqueIDs)

    def generateMissingImages(self, folder):
        self.missingImageNames = []
        self.missingImageNumbers = []
        for name in self.allNames:
            if not os.path.isfile(folder + "/" + name + ".gif"):
                self.missingImageNames.append(name)
                self.missingImageNumbers.append(self.nameToNumber(name))
        if len(self.missingImageNames) > 0:
            print("Warning: can't find", len(self.missingImageNames), "images!")

    def generateNumberToTypes(self):
        self.numberToTypesDict = {}
        self.numberToPrimaryTypeDict = {}
        self.numberToSecondaryTypeDict = {}
        self.uniqueValidIDs = []
        self.uniqueValidPokemonNames = []
        with open(self.csv_path + "pokemon_types.csv") as f:
            reader = csv.DictReader(f)
            for row in reader:
                pokemon_id = int(row['pokemon_id'])
                type_id = int(row['type_id'])
                if pokemon_id in self.missingImageNumbers:
                    continue  # ignore pokemon that we don't have images for
                if row['slot'] == '1':
                    self.uniqueValidIDs.append(pokemon_id)
                    self.uniqueValidPokemonNames.append(self.numberToName(pokemon_id))
                    self.numberToTypesDict[pokemon_id] = [type_id]
                    self.numberToPrimaryTypeDict[pokemon_id] = type_id
                    self.numberToSecondaryTypeDict[pokemon_id] = -1
                else:
                    self.numberToTypesDict[pokemon_id].append(type_id)
                    self.numberToSecondaryTypeDict[pokemon_id] = type_id

        # By now, we know which pokemon we will use
        self.uniqueValidIDsArray = np.array(self.uniqueValidIDs)
        self.PrimaryTypesArray   = np.array([[n, self.numberToPrimaryTypeDict[n]] for n in self.uniqueValidIDs])
        self.SecondaryTypesArray = np.array([[n, self.numberToSecondaryTypeDict[n]] for n in self.uniqueValidIDs])

    def generateTypeNames(self):
        self.type_names = []
        self.typeToNameDict = {}
        self.nameToTypeDict = {}
        with open(self.csv_path + "type_names.csv") as f:
            reader = csv.DictReader(f)
            for row in reader:
                name = self.normalizeString(row['name'])
                number = int(row['type_id'])
                if row['local_language_id'] == '9':
                    if number > 18:
                        return
                    self.typeToNameDict[number] = name
                    self.nameToTypeDict[name] = number
                    self.type_names.append(name)

    def generateTrainValTestSplit(self):
        try:
            train = np.loadtxt(self.splits_path + "Train.txt", dtype=int)
            val   = np.loadtxt(self.splits_path + "Val.txt", dtype=int)
            test  = np.loadtxt(self.splits_path + "Test.txt", dtype=int)
            self.x_train = train[:,0]
            self.x_val   = val[:,0]
            self.x_test  = test[:,0]
            self.y_train = train[:,1]
            self.y_val   = val[:,1]
            self.y_test  = test[:,1]
        except:
            print("Error: couldn't load train/val/test splits")

    def normalizeString(self, s):
        return s.lower()

    def getTypeSample(self, type, k=5, split='all'):
        sample = None
        isType = None
        if split == 'train':
            isType = self.y_train == type
            sample = self.x_train[isType]
        elif split == 'val':
            isType = self.y_val == type
            sample = self.x_val[isType]
        elif split == 'test':
            isType = self.y_test == type
            sample = self.x_test[isType]
        else:
            isType = self.PrimaryTypesArray[:,1] == type
            sample = self.PrimaryTypesArray[isType,0]

        k = min(k, sample.shape[0])
        if k > 0:
            return np.random.choice(sample, k, replace=False)
        else:
            return np.array([])

    def generateTypeQuizHTML(self, k=5, q=10):
        html = "<!DOCTYPE html><html>"
        html += "<title>Type Quiz</title>"
        # html += "<script type='text/javascript'</script>"
        html += "<script src='TypeQuiz.js'></script>"
        html += "<body>"

        # Add class examples
        html += "<h2>Type Examples</h2>"
        for i in range(18):
            type_id = i+1
            html += "<h3>" + self.typeToName(type_id).capitalize() + "</h3>"
            for pkmn in self.numbersToName(self.getTypeSample(type_id, k, 'train')):
                html += '<img src="' + self.gifs_path + pkmn + '.gif">'

        optionTags = ""
        for i in range(18):
            optionTags += "<option>" + self.typeToName(i+1).capitalize() + "</option>"


        # Add quiz portion
        html += "<h2>Type Quiz</h2>"
        indices = np.random.choice(np.arange(self.x_test.shape[0]), q, replace=False)
        for i in range(q):
            index = indices[i]
            x = self.x_test[index]
            y = self.y_test[index]
            pkmn = self.numberToName(x)
            html += "<div style='margin-bottom:15px' id='question" + str(i) + "'>"
            html += '<img src="' + self.gifs_path + str(pkmn) + '.gif">'
            html += "<select>"
            html += optionTags
            html += "</select>"
            html += "<span style='visibility:hidden; margin-left:10px;'>" + self.typeToName(y).capitalize() + "</span>"
            html += "</div>"

        # Add submit button
        html += "<button id='check-quiz'>Check Quiz!</button>"
        html += "<div id='result'></div>"

        html += "</body><html>"
        f = open("TypeSample.html","w")
        f.write(html)
        f.close()

    def typeToName(self, n):
        return self.typeToNameDict[n]

    def nameToType(self, name):
        return self.nameToTypeDict[self.normalizeString(name)]

    def numberToTypes(self, n):
        return self.numberToTypesDict[n]

    def numberToPrimaryType(self, n):
        return self.numberToPrimaryTypeDict[n]

    def numberToSecondaryType(self, n):
        return self.numberToSecondaryTypeDict[n]

    def numberToName(self, n):
        return self.numberToNameDict[n]

    def numbersToName(self, l):
        return [self.numberToName(p) for p in l]

    def nameToNumber(self, name):
        return self.nameToNumberDict[self.normalizeString(name)]

    # from https://stackoverflow.com/questions/50054187/convert-animated-gif-to-4d-array-in-python
    # GIF path name --> 4D numpy array
    def readGif(self, path, format='RGBA'):
        channels = 4
        if format != 'RGBA':
            channels = 3

        img = Image.open(path)
        gif = np.array([np.array(frame.copy().convert(format).getdata(),dtype=np.uint8).reshape(frame.size[1],frame.size[0],channels) for frame in ImageSequence.Iterator(img)])
        return gif

if __name__ == "__main__":
    utils = PokemonUtils()

    # Find distribution of data sizes
    smallest = np.array([999,999,999,999])
    largest  = np.array([-1,-1,-1,-1])
    count = 1;
    for name in utils.uniqueValidPokemonNames:
        gif = utils.readGif("../Data/pkparaiso/" + name + ".gif")
        cur = np.array([gif.shape[0], gif.shape[1], gif.shape[2], gif.shape[3]])
        smallest = np.minimum(cur, smallest)
        largest  = np.maximum(cur, largest)
        if count % 50 == 0:
            print(count, "done!")
        count += 1
    print("smallest:", smallest)
    print("largest:", largest)

    # Check readGif functionality
    # gif = utils.readGif("../Data/pkparaiso/araquanid.gif")
    # for i in range(3):
    #     plt.imshow(gif[i,:,:,:])
    #     plt.show()

    # Check flying-type distribution
    # print(utils.numbersToName(utils.getTypeSample(utils.nameToType('flying'), 100, 'train')))
    # print(utils.numbersToName(utils.getTypeSample(utils.nameToType('flying'), 100, 'val')))
    # print(utils.numbersToName(utils.getTypeSample(utils.nameToType('flying'), 100, 'test')))

    # Generate Type Quiz
    # utils.generateTypeQuizHTML(k=10, q=20)
