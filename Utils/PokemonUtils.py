import csv
import numpy as np
import os.path

class PokemonUtils():

    def __init__(self, verbose=True):
        self.path = "pokedex-master/pokedex/data/csv/"
        self.gifsPath = "pkparaiso"
        self.verbose = verbose
        self.generateNameToNumber()  # loads all names/numbers from veekun file
        self.generateMissingImages(self.gifsPath)  # cross-references with downloaded images
        self.generateNumberToTypes()  # Only for pokemon that have associated images
        self.generateTypeNames()  # Only for pokemon that have associated images
        self.generateTrainValTestSplit(0.8, 0.8)

    def normalizeString(self, s):
        return s.lower()

    def generateTypeNames(self):
        self.typeToNameDict = {}
        self.nameToTypeDict = {}
        with open(self.path + "type_names.csv") as f:
            reader = csv.DictReader(f)
            for row in reader:
                name = self.normalizeString(row['name'])
                number = int(row['type_id'])
                if row['local_language_id'] == '9':
                    self.typeToNameDict[number] = name
                    self.nameToTypeDict[name] = number

    def generateNumberToTypes(self):
        self.numberToTypesDict = {}
        self.numberToPrimaryTypeDict = {}
        self.numberToSecondaryTypeDict = {}
        self.uniqueValidIDs = []
        with open(self.path + "pokemon_types.csv") as f:
            reader = csv.DictReader(f)
            for row in reader:
                pokemon_id = int(row['pokemon_id'])
                type_id = int(row['type_id'])
                if pokemon_id in self.missingImageNumbers:
                    continue  # ignore pokemon that we don't have images for
                if row['slot'] == '1':
                    self.uniqueValidIDs.append(pokemon_id)
                    self.numberToTypesDict[pokemon_id] = [type_id]
                    self.numberToPrimaryTypeDict[pokemon_id] = type_id
                    self.numberToSecondaryTypeDict[pokemon_id] = -1
                else:
                    self.numberToTypesDict[pokemon_id].append(type_id)
                    self.numberToSecondaryTypeDict[pokemon_id] = type_id

        self.uniqueValidIDsArray = np.array(self.uniqueValidIDs)
        self.PrimaryTypesArray   = np.array([[n, self.numberToPrimaryTypeDict[n]] for n in self.uniqueValidIDs])
        self.SecondaryTypesArray = np.array([[n, self.numberToSecondaryTypeDict[n]] for n in self.uniqueValidIDs])

    def generateNameToNumber(self):
        self.nameToNumberDict = {}
        self.numberToNameDict = {}
        self.uniqueIDs = []
        with open(self.path + "pokemon.csv") as f:
            reader = csv.DictReader(f)
            for row in reader:
                id = int(row['id'])
                name = self.normalizeString(row['identifier'])
                self.uniqueIDs.append(id)
                self.nameToNumberDict[name] = id
                self.numberToNameDict[id] = name

        self.allNames = self.numbersToName(self.uniqueIDs)

    def generateTrainValTestSplit(self, train, dev):
        n = self.PrimaryTypesArray.shape[0]
        inds = np.arange(n)
        np.random.shuffle(inds)
        train_inds = inds[:int(n*train)]
        val_inds = inds[int(n*train):int(n*dev)]
        test_inds = inds[int(n*dev):]
        self.x_train, self.x_val, self.x_test = self.PrimaryTypesArray[train_inds,0], self.PrimaryTypesArray[val_inds,0], self.PrimaryTypesArray[test_inds,0]
        self.y_train, self.y_val, self.y_test = self.PrimaryTypesArray[train_inds,1], self.PrimaryTypesArray[val_inds,1], self.PrimaryTypesArray[test_inds,1]

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
                html += '<img src="' + self.gifsPath + "/" + pkmn + '.gif">'

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
            html += '<img src="' + self.gifsPath + "/" + str(pkmn) + '.gif">'
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

    def generateMissingImages(self, folder):
        self.missingImageNames = []
        self.missingImageNumbers = []
        for name in self.allNames:
            if not os.path.isfile(folder + "/" + name + ".gif"):
                self.missingImageNames.append(name)
                self.missingImageNumbers.append(self.nameToNumber(name))
        if len(self.missingImageNames) > 0:
            print("Warning: can't find", len(self.missingImageNames), "images!")

if __name__ == "__main__":
    utils = PokemonUtils()
    print(utils.numbersToName(utils.getTypeSample(utils.nameToType('flying'), 100, 'train')))
    print(utils.numbersToName(utils.getTypeSample(utils.nameToType('flying'), 100, 'val')))
    print(utils.numbersToName(utils.getTypeSample(utils.nameToType('flying'), 100, 'test')))
    utils.generateTypeQuizHTML(k=10, q=20)
