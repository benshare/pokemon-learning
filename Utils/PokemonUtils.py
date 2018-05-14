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
* utils.generateXYSplitsV1()
Make x_train, x_val, and x_test from first frame of every gifs
Returns (x_train, x_val, x_test, y_train, y_val, y_test)
  X Shapes: (N, H, W, C)
  Y Shapes: (N,)

* utils.x_train_inds, .x_val_inds, .x_test_inds
(N,) arrays of training indices

* utils.x_train_pokedex, .x_val_pokedex, .x_test_pokedex
(N,) arrays of pokedex numbers

* .y_train, .y_val, .y_test
(N,) arrays of type numbers (1 = Normal, ..., 18 = Fairy)


GIF MANIPULATION:
-----------------
* utils.loadAllGIFs(verbose=True)
Loads all relevant GIFs into RAM.
Takes ~5 minutes!

* utils.readGif(path)
gif path name --> numpy array (T, H, W, C)


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

    def __init__(self, verbose=True, utils=None):
        self.csv_path = "../Data/veekun/"
        self.gifs_path = "../Data/pkparaiso/"
        self.splits_path = "../Data/Splits/"
        self.all_gifs = []
        self.verbose = verbose

        self.generateNameToNumber()  # loads all names/numbers from veekun file
        self.generateMissingImages(self.gifs_path)  # cross-references with downloaded images
        self.generateNumberToTypes()  # Only for pokemon that have associated images
        self.generateTypeNames()  # Only for pokemon that have associated images
        self.generateTrainValTestSplit()  # Load from file

        if not utils is None:
            self.all_gifs = utils.all_gifs
            print("loaded previous", len(self.all_gifs), "datapoints")
            self.calculateGIFSizeExtremes()
            self.data_loaded = True
        else:
            self.data_loaded = False

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
            self.x_train_inds = train[:,0]
            self.x_val_inds   = val[:,0]
            self.x_test_inds  = test[:,0]
            self.x_train_pokedex = train[:,1]
            self.x_val_pokedex   = val[:,1]
            self.x_test_pokedex  = test[:,1]
            self.y_train = train[:,2]
            self.y_val   = val[:,2]
            self.y_test  = test[:,2]
        except:
            print("Error: couldn't load train/val/test splits")

    def normalizeString(self, s):
        return s.lower()

    def getTypeSample(self, type, k=5, split='all'):
        sample = None
        isType = None
        if split == 'train':
            isType = self.y_train == type
            sample = self.x_train_pokedex[isType]
        elif split == 'val':
            isType = self.y_val == type
            sample = self.x_val_pokedex[isType]
        elif split == 'test':
            isType = self.y_test == type
            sample = self.x_test_pokedex[isType]
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
        indices = np.random.choice(np.arange(self.x_test_pokedex.shape[0]), q, replace=False)
        for i in range(q):
            index = indices[i]
            x = self.x_test_pokedex[index]
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
    def readGif(self, name, folder="", format='RGBA'):
        if folder == "":
            folder = self.gifs_path

        channels = 4
        if format != 'RGBA':
            channels = 3

        img = Image.open(folder + name + ".gif")
        gif = np.array([np.array(frame.copy().convert(format).getdata(),dtype=np.uint8).reshape(frame.size[1],frame.size[0],channels) for frame in ImageSequence.Iterator(img)])
        return gif

    # Collect list of all gifs (takes ~5 minutes)
    def loadAllGIFs(self):
        self.all_gifs = []
        count = 1;
        for name in self.uniqueValidPokemonNames:
            gif = self.readGif(name)
            self.all_gifs.append(gif)
            if self.verbose and count % 25 == 0:
                print(count, "done!")
            count += 1
        print("all", count-1, "done!")
        self.calculateGIFSizeExtremes()
        self.data_loaded = True

    def calculateGIFSizeExtremes(self):
        # Find distribution of data sizes
        self.smallest_image_size = np.array([999,999,999,999])
        self.largest_image_size  = np.array([-1,-1,-1,-1])
        for i in range(len(self.all_gifs)):
            gif = self.all_gifs[i]
            cur = np.array([gif.shape[0], gif.shape[1], gif.shape[2], gif.shape[3]])
            if (gif.shape[0] <= 1):
                print(self.numberToName(self.uniqueValidIDs[i]), "had only 1 frame")
                continue
            self.smallest_image_size = np.minimum(cur, self.smallest_image_size)
            self.largest_image_size  = np.maximum(cur, self.largest_image_size)
        print("smallest:", self.smallest_image_size)
        print("largest:", self.largest_image_size)

    # Make x_train, x_val, and x_test from first frame of every gifs
    # Returns (x_train, x_val, x_test, y_train, y_val, y_test)
    # X Shapes: (N, H, W, C)
    # Y Shapes: (N,)
    def generateXYSplitsV1(self):
        if not self.data_loaded:
            print("Error: load data first!")
            return

        largest = self.largest_image_size
        x_train = np.zeros((self.x_train_inds.shape[0], largest[1], largest[2], largest[3]), dtype='uint8')
        x_val   = np.zeros((self.x_val_inds.shape[0],   largest[1], largest[2], largest[3]), dtype='uint8')
        x_test  = np.zeros((self.x_test_inds.shape[0],  largest[1], largest[2], largest[3]), dtype='uint8')

        mats = (x_train, x_val, x_test)
        inds = (self.x_train_inds, self.x_val_inds, self.x_test_inds)
        for i in range(3):
            for n in range(mats[i].shape[0]):
                frame = self.all_gifs[inds[i][n]][0,:,:,:]
                h_diff = largest[1] - frame.shape[0]
                w_diff = largest[2] - frame.shape[1]
                h0 = h_diff//2
                h1 = h_diff//2 + h_diff%2
                w0 = w_diff//2
                w1 = w_diff//2 + w_diff%2
                padded = np.pad(frame, ((h0,h1),(w0,w1),(0,0)), 'constant')  # Centered on transparent bkgd
                mats[i][n,:,:,:] = padded

        y_train = self.y_train
        y_val   = self.y_val
        y_test  = self.y_test

        if self.verbose:
            print("x_train:", x_train.shape)
            print("x_val:  ", x_val.shape)
            print("x_test: ", x_test.shape)
            print("y_train:", y_train.shape)
            print("y_val:  ", y_val.shape)
            print("y_test: ", y_test.shape)

        return (x_train, x_val, x_test, y_train, y_val, y_test)

if __name__ == "__main__":
    utils = PokemonUtils()

    # Find distribution of data sizes
    # smallest = np.array([999,999,999,999])
    # largest  = np.array([-1,-1,-1,-1])
    # count = 1;
    # for name in utils.uniqueValidPokemonNames:
    #     gif = utils.readGif("../Data/pkparaiso/" + name + ".gif")
    #     cur = np.array([gif.shape[0], gif.shape[1], gif.shape[2], gif.shape[3]])
    #     smallest = np.minimum(cur, smallest)
    #     largest  = np.maximum(cur, largest)
    #     if count % 50 == 0:
    #         print(count, "done!")
    #     count += 1
    # print("smallest:", smallest)
    # print("largest:", largest)

    # Check readGif functionality
    gif = utils.readGif("araquanid")
    for i in range(3):
        plt.imshow(gif[i,:,:,:])
        plt.show()

    # Check flying-type distribution
    # print(utils.numbersToName(utils.getTypeSample(utils.nameToType('flying'), 100, 'train')))
    # print(utils.numbersToName(utils.getTypeSample(utils.nameToType('flying'), 100, 'val')))
    # print(utils.numbersToName(utils.getTypeSample(utils.nameToType('flying'), 100, 'test')))

    # Generate Type Quiz
    # utils.generateTypeQuizHTML(k=500, q=20)
