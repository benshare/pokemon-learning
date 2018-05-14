from PokemonUtils import *
import matplotlib.pyplot as plt

if __name__ == "__main__":
    save_folder = "../Data/Splits/"
    train_percent, dev_percent = 0.64, 0.80;  # 80/20 (train+val)/test, 80/20 train/val

    utils = PokemonUtils()
    n = utils.PrimaryTypesArray.shape[0]
    inds = np.arange(n)
    np.random.shuffle(inds)
    train_inds = inds[:int(n*train_percent)]
    val_inds = inds[int(n*train_percent):int(n*dev_percent)]
    test_inds = inds[int(n*dev_percent):]

    train, val, test = (
        np.concatenate((train_inds.reshape(-1,1), utils.PrimaryTypesArray[train_inds,:]), axis=1),
        np.concatenate((val_inds.reshape(-1,1), utils.PrimaryTypesArray[val_inds,:]), axis=1),
        np.concatenate((test_inds.reshape(-1,1), utils.PrimaryTypesArray[test_inds,:]), axis=1))

    print("train: " + str(train.shape[0]))
    print("dev:   " + str(val.shape[0]))
    print("test:  " + str(test.shape[0]))

    split_names = ("Train", "Val", "Test")
    splits = (train, val, test)
    for i in range(3):
        fig, ax = plt.subplots()
        ax.hist(splits[i][:,2], np.arange(20), rwidth=0.75)
        ax.set_xticks(np.arange(18)+1.5)
        ax.set_xticklabels(utils.type_names, rotation=45, rotation_mode="anchor", ha="right")
        plt.title = split_names[i] + " Distribution"
        plt.savefig(save_folder + split_names[i] + "Distribution.png")

        np.savetxt(save_folder + split_names[i] + '.txt', splits[i], fmt='%i')
