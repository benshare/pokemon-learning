import csv
import numpy as np
import matplotlib.pyplot as plt
import os.path  # File path checking
from PIL import Image, ImageSequence  # Reading GIFs
import os

"""
PokemonUtils
------------
* Useage: utils = PokemonUtils()


DATA:
-----
* utils.localLoadAllGifs(filename)
Loads train/val/test data from a .npz file generated from generateXYSplits or some variant

* utils.generateXYSplitsV1(H=128, W=128, C=4)
Make x_train, x_val, and x_test from first frame of every gifs
Returns (x_train, x_val, x_test, y_train, y_val, y_test)
  X Shapes: (N, H, W, C)
  Y Shapes: (N,)

* utils.generateXYSplitsV2(K=10, H=128, W=128, C=4, outfile=None)
Make x_train, x_val, and x_test from a randomly chosen K frames from every GIF
Returns (x_train, x_val, x_test, y_train, y_val, y_test)
  X Shapes: (N*K, H, W, C)
  Y Shapes: (N*K,)

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

    def __init__(self, rel_loc="../", verbose=True, utils=None):
        self.rel_loc = rel_loc
        self.csv_path = rel_loc + "Data/veekun/"
        self.gifs_path = rel_loc + "Data/pkparaiso/"
        self.splits_path = rel_loc + "Data/Splits/"
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
        self.typeNameToColorDict = {
            "normal":"#a8a77a",
            "fighting":"#bf2f2b",
            "poison":"#9e449e",
            "ground":"#e9be6c",
            "flying":"#a793ee",
            "bug":"#a9b62f",
            "rock":"#b89e3f",
            "ghost":"#6f5a97",
            "steel":"#b8b9cf",
            "fire":"#ef7e37",
            "water":"#6794ee",
            "electric":"#f8ce3f",
            "grass":"#7bc656",
            "ice":"#99d8d8",
            "psychic":"#f65988",
            "dragon":"#6d47f5",
            "dark":"#705849",
            "fairy":"#feb9f9",
        }

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
        # self.calculateGIFSizeExtremes()
        self.data_loaded = True

    def localLoadAllGifs(self, path):
        files = np.load(self.rel_loc + path)
        self.x_train = files['arr_0']
        self.x_val   = files['arr_1']
        self.x_test  = files['arr_2']
        self.y_train = files['arr_3']
        self.y_val   = files['arr_4']
        self.y_test  = files['arr_5']

        self.printSplitSizes()

    # Get largest and smallest value for each dimension
    # (T, H, W, C)
    def calculateGIFSizeExtremes(self):
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

    def fitImageToSize(self, im, h=128, w=128):
        h_diff = h - im.shape[0]
        w_diff = w - im.shape[1]

        sizedH = None
        sizedHW = None

        # Height
        if (h_diff >= 0):
            h0 = h_diff//2
            h1 = h_diff//2 + h_diff%2
            sizedH = np.pad(im, ((h0,h1),(0,0),(0,0)), 'constant')  # Centered on transparent bkgd
        else:
            offset = np.abs(h_diff) // 2  #np.random.randint(np.abs(h_diff))
            sizedH = im[offset:h+offset, :, :]

        # Width
        if (w_diff >= 0):
            w0 = w_diff//2
            w1 = w_diff//2 + w_diff%2
            sizedHW = np.pad(sizedH, ((0,0),(w0,w1),(0,0)), 'constant')  # Centered on transparent bkgd
        else:
            offset = np.abs(w_diff) // 2  #np.random.randint(np.abs(w_diff))
            sizedHW = sizedH[:, offset:w+offset, :]

        return sizedHW


    # Make x_train, x_val, and x_test from first frame of every gifs
    # X Shapes: (N, H, W, C)
    # Y Shapes: (N,)
    def generateXYSplitsV1(self, H=128, W=128, C=4):
        if not self.data_loaded:
            print("Error: load data first!")
            return

        x_train = np.zeros((self.x_train_inds.shape[0], H, W, C), dtype='uint8')
        x_val   = np.zeros((self.x_val_inds.shape[0],   H, W, C), dtype='uint8')
        x_test  = np.zeros((self.x_test_inds.shape[0],  H, W, C), dtype='uint8')

        mats = (x_train, x_val, x_test)
        inds = (self.x_train_inds, self.x_val_inds, self.x_test_inds)
        for i in range(3):
            for n in range(mats[i].shape[0]):
                frame = self.all_gifs[inds[i][n]][0,:,:,:]
                mats[i][n,:,:,:] = self.fitImageToSize(frame, H, W)

        y_train = self.y_train
        y_val   = self.y_val
        y_test  = self.y_test

        self.printSplitSizes()

    def printSplitSizes(self):
        if self.verbose:
            print("x_train:", self.x_train.shape)
            print("x_val:  ", self.x_val.shape)
            print("x_test: ", self.x_test.shape)
            print("y_train:", self.y_train.shape)
            print("y_val:  ", self.y_val.shape)
            print("y_test: ", self.y_test.shape)

    # Make x_train, x_val, and x_test from a randomly chosen k frames from every GIF
    # X Shapes: (N*K, H, W, C)
    # Y Shapes: (N*K,)
    def generateXYSplitsV2(self, K=10, H=128, W=128, C=4, outFile=None):
        if not self.data_loaded:
            print("Error: load data first!")
            return

        self.calculateGIFSizeExtremes()
        assert(K <= self.smallest_image_size[0])  # Can't take more frames than the shortest GIF has

        x_train = np.zeros((self.x_train_inds.shape[0]*K, H, W, C), dtype='uint8')
        x_val   = np.zeros((self.x_val_inds.shape[0]*K,   H, W, C), dtype='uint8')
        x_test  = np.zeros((self.x_test_inds.shape[0]*K,  H, W, C), dtype='uint8')

        mats = (x_train, x_val, x_test)
        inds = (self.x_train_inds, self.x_val_inds, self.x_test_inds)
        for i in range(3):
            for n in range(mats[i].shape[0]//K):
                T = self.all_gifs[inds[i][n]].shape[0]
                frames = np.random.choice(np.arange(T), K, replace=False)
                framesAdded = 0
                for k in frames:
                    frame = self.all_gifs[inds[i][n]][k,:,:,:]
                    mats[i][K*n+framesAdded,:,:,:] = self.fitImageToSize(frame, H, W)
                    framesAdded += 1

        y_train = np.repeat(self.y_train, K)
        y_val   = np.repeat(self.y_val, K)
        y_test  = np.repeat(self.y_test, K)

        if self.verbose:
            print("x_train:", x_train.shape)
            print("x_val:  ", x_val.shape)
            print("x_test: ", x_test.shape)
            print("y_train:", y_train.shape)
            print("y_val:  ", y_val.shape)
            print("y_test: ", y_test.shape)

        if not outFile is None:
            np.savez(self.rel_loc + "Data/SplitsV2", x_train, x_val, x_test, y_train, y_val, y_test)

        self.x_train, self.x_val, self.x_test, self.y_train, self.y_val, self.y_test = x_train, x_val, x_test, y_train, y_val, y_test
        
    # Make y_train_2, y_val_2, and y_test_2
    # Y Shapes: (N*K,)
    def getSecondaryTypeLabels(self, K=10, outFile=None):
        mats = [None] * 3
        inds = (self.x_train_inds, self.x_val_inds, self.x_test_inds)
        for i in range(3):
            mats[i] = self.SecondaryTypesArray[:,1][inds[i]]
            
        y_train_2 = np.repeat(mats[0], K)
        y_val_2   = np.repeat(mats[1], K)
        y_test_2  = np.repeat(mats[2], K)

        if not outFile is None:
            np.savez(self.rel_loc + "Data/SecondaryTypeLabels", y_train_2, y_val_2, y_test_2)
            
        if self.verbose:
            print("y_train_2:", y_train_2.shape)
            print("y_val_2:  ", y_val_2.shape)
            print("y_test_2: ", y_test_2.shape)

        self.y_train_2, self.y_val_2, self.y_test_2 = y_train_2, y_val_2, y_test_2
        return (y_train_2, y_val_2, y_test_2)
                
        
    def getSplits(self):
        return (self.x_train, self.x_val, self.x_test, self.y_train, self.y_val, self.y_test)

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
    # gif = utils.readGif("araquanid")
    # for i in range(3):
    #     plt.imshow(gif[i,:,:,:])
    #     plt.show()

    # Check flying-type distribution
    # print(utils.numbersToName(utils.getTypeSample(utils.nameToType('flying'), 100, 'train')))
    # print(utils.numbersToName(utils.getTypeSample(utils.nameToType('flying'), 100, 'val')))
    # print(utils.numbersToName(utils.getTypeSample(utils.nameToType('flying'), 100, 'test')))

    # View sorted disstribution of types
    # # 1: sort data
    # data = np.bincount(utils.PrimaryTypesArray[:,1])[1:]
    # sorted_data = np.sort(data)
    # order = np.argsort(data)
    # sorted_labels = [utils.type_names[i].capitalize() for i in order]
    # label_colors = [utils.typeNameToColorDict[type.lower()] for type in sorted_labels]
    # # 2: generate bar graph
    # f, ax = plt.subplots()
    # ax.set_title("Distribution of Primary Types")
    # d = np.diff(np.unique(data)).min()
    # left_of_first_bin = data.min() - float(d)/2
    # right_of_last_bin = data.max() + float(d)/2
    # ax.bar(np.arange(18)+1, sorted_data, color=label_colors)
    # ax.set_xticks(np.arange(18)+1)
    # ax.set_xticklabels(sorted_labels, rotation=45, rotation_mode="anchor", ha="right")
    # plt.show()

    # View disstribution of types
    # f, ax = plt.subplots()
    # ax.set_title("Distribution of Primary Types")
    # data = utils.PrimaryTypesArray[:,1]
    # d = np.diff(np.unique(data)).min()
    # left_of_first_bin = data.min() - float(d)/2
    # right_of_last_bin = data.max() + float(d)/2
    # ax.hist(data, np.arange(left_of_first_bin, right_of_last_bin + d, d), rwidth=0.75)
    # ax.set_xticks(np.arange(18)+1)
    # ax.set_xticklabels(utils.type_names, rotation=45, rotation_mode="anchor", ha="right")
    # plt.show()

    # Generate Type Quiz
    # utils.generateTypeQuizHTML(k=5, q=20)
