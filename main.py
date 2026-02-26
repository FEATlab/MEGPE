# python packages
import argparse
import random
import time
import operator

import evalGP_main as evalGP
# only for strongly typed GP
import gp_restrict
import numpy as np
import select_nsga
from deap import base, creator, tools, gp
from strongGPDataType import Int1, Int2, Int3, Float1, Float2, Float3, Img, Img1, Vector, Vector1
import feature_function as fe_fs
from sklearn.svm import LinearSVC
from sklearn.metrics import confusion_matrix, classification_report
from sklearn import preprocessing
from sklearn.model_selection import StratifiedKFold
import sys
from data.ImbalanceCIFAR import IMBALANCECIFAR10,IMBALANCEFMNIST,IMBALANCEMNIST,IMBALANCESTL10,imbalance_MRBand
import warnings
import pickle
from sklearn.metrics import accuracy_score

warnings.simplefilter("ignore")
'FLGP_MEGPE'

parser = argparse.ArgumentParser()
parser.add_argument(
    "--log_file", default="log/main.log", help="log file name", type=str
)
parser.add_argument(
    "--pkl_file", help="pkl file name", default="log/latest_population.pkl"
)
parser.add_argument(
    "--dataset",
    help="Dataset name (choose from the provided options)",
    type=str,
    choices=["mnist", "cifar10", "fashion_mnist", "STL", "MBI", "MRB","MRD"],  # 限制可选值
    default="MRD"
)
parser.add_argument(
    "--randomSeeds", default=1
)

opt = parser.parse_args()


# Logger class remains unchanged
class Logger(object):
    def __init__(self, filename='default.log', add_flag=True, stream=sys.stdout):
        self.terminal = stream
        print("filename:", filename)
        self.filename = filename
        self.add_flag = add_flag
        # self.log = open(filename, 'a+')

    def write(self, message):
        if self.add_flag:
            with open(self.filename, 'a+') as log:
                self.terminal.write(message)
                log.write(message)
        else:
            with open(self.filename, 'w') as log:
                self.terminal.write(message)
                log.write(message)

    def flush(self):
        pass

def create_dataset(dataset):
    if dataset=="cifar10":
        train_dataset = IMBALANCECIFAR10("train", imbalance_ratio=100, root='./data/CIFAR-10')
        test_dataset = IMBALANCECIFAR10("test", imbalance_ratio=100, root='./data/CIFAR-10')
        Is_RGB=True
        Flag=1
    if Flag==1:
        x_train = train_dataset.data
        y_train = np.array(train_dataset.labels)
        x_test = test_dataset.data
        if Is_RGB==False:
            x_test = x_test.numpy()
        y_test = np.array(test_dataset.labels)
        x_train = x_train.astype(np.float32)
        x_test = x_test.astype(np.float32)
        dim=x_train.ndim
        if dim==4:
            shape=x_train.shape
            if shape[1] !=3:
                dims = list(range(dim))
                dims[-1], dims[1] = dims[1], dims[-1]
                x_train = x_train.transpose(dims)
                x_test = x_test.transpose(dims)
    return x_train,y_train,x_test,y_test,Is_RGB


x_train,y_train,x_test,y_test,Is_RGB = create_dataset(opt.dataset)

unique_classes, counts = np.unique(y_train, return_counts=True)


group1_indices = []
group2_indices = []
group3_indices = []

for label, count in zip(unique_classes, counts):
    if count >= 100:
        group1_indices.append(label)
    elif 50 <= count < 100:
        group2_indices.append(label)
    else:
        group3_indices.append(label)

print('x_train.shape:', x_train.shape)
print('x_test.shape:', x_test.shape)

unique_labels = np.unique(y_train)
num_unique_labels = len(unique_labels)
total_instances = len(y_train)
per_class_num = [np.sum(y_train == label) for label in unique_labels]
class_ratios = [np.sum(y_train == label) / total_instances for label in unique_labels]

class_reciprocals = [1 / num for num in per_class_num]
sum_of_reciprocals = sum(class_reciprocals)

# Parameters
pop_size = 100
generation = 50
cxProb = 0.8
mutProb = 0.2
totalRuns = 1
initialMinDepth = 4
initialMaxDepth = 15
maxDepth = 15


## GP

def make_Sigma():
    return random.randint(1, 4)


def make_Order():
    return random.randint(0, 3)


def make_Theta():
    return random.randint(0, 8)


def make_Frequency():
    return random.randint(0, 5)


def make_n():
    return round(random.random(), 3)


def make_KernelSize():
    return random.randrange(2, 5, 2)


def make_classificationN():
    return random.randint(1, 4)

if Is_RGB:
    pset = gp.PrimitiveSetTyped('MAIN', [Img, Img, Img], Vector1, prefix='Image')
else:
    pset = gp.PrimitiveSetTyped('MAIN', [Img], Vector1, prefix='Image')
# image classification layer
# pset.addPrimitive(fe_fs.classifier_selection, [Vector1, Int4], ClassLabel, name='classifier')

# Feature concatenation
pset.addPrimitive(fe_fs.root_con, [Vector1, Vector1], Vector1, name='Root')
pset.addPrimitive(fe_fs.root_conVector2, [Img1, Img1], Vector1, name='Root2')
pset.addPrimitive(fe_fs.root_conVector3, [Img1, Img1, Img1], Vector1, name='Root3')
pset.addPrimitive(fe_fs.root_con, [Vector, Vector], Vector1, name='Roots2')
pset.addPrimitive(fe_fs.root_con, [Vector, Vector, Vector], Vector1, name='Roots3')
pset.addPrimitive(fe_fs.root_con, [Vector, Vector, Vector, Vector], Vector1, name='Roots4')

# Feature extraction
pset.addPrimitive(fe_fs.global_hog_small, [Img1], Vector, name='Global_HOG')
pset.addPrimitive(fe_fs.all_lbp, [Img1], Vector, name='Global_uLBP')
pset.addPrimitive(fe_fs.all_sift, [Img1], Vector, name='Global_SIFT')
pset.addPrimitive(fe_fs.global_hog_small, [Img], Vector, name='FGlobal_HOG')
pset.addPrimitive(fe_fs.all_lbp, [Img], Vector, name='FGlobal_uLBP')
pset.addPrimitive(fe_fs.all_sift, [Img], Vector, name='FGlobal_SIFT')

# Pooling
pset.addPrimitive(fe_fs.maxP, [Img1, Int3, Int3], Img1, name='MaxPF')

# Filtering (输入都是图片，输出也是图片，并且大小没有改变)
pset.addPrimitive(fe_fs.gau, [Img1, Int1], Img1, name='GauF')
pset.addPrimitive(fe_fs.gauD, [Img1, Int1, Int2, Int2], Img1, name='GauDF')
pset.addPrimitive(fe_fs.gab, [Img1, Float1, Float2], Img1, name='GaborF')
pset.addPrimitive(fe_fs.laplace, [Img1], Img1, name='LapF')
pset.addPrimitive(fe_fs.gaussian_Laplace1, [Img1], Img1, name='LoG1F')
pset.addPrimitive(fe_fs.gaussian_Laplace2, [Img1], Img1, name='LoG2F')
pset.addPrimitive(fe_fs.sobelxy, [Img1], Img1, name='SobelF')
pset.addPrimitive(fe_fs.sobelx, [Img1], Img1, name='SobelXF')
pset.addPrimitive(fe_fs.sobely, [Img1], Img1, name='SobelYF')
pset.addPrimitive(fe_fs.medianf, [Img1], Img1, name='MedF')
pset.addPrimitive(fe_fs.meanf, [Img1], Img1, name='MeanF')
pset.addPrimitive(fe_fs.minf, [Img1], Img1, name='MinF')
pset.addPrimitive(fe_fs.maxf, [Img1], Img1, name='MaxF')
pset.addPrimitive(fe_fs.lbp, [Img1], Img1, name='LBPF')
pset.addPrimitive(fe_fs.hog_feature, [Img1], Img1, name='HoGF')
pset.addPrimitive(fe_fs.sqrt, [Img1], Img1, name='SqrtF')
pset.addPrimitive(fe_fs.relu, [Img1], Img1, name='ReLUF')

# Pooling for Img
pset.addPrimitive(fe_fs.maxP, [Img, Int3, Int3], Img1, name='MaxP')

# Filtering for Img
pset.addPrimitive(fe_fs.gau, [Img, Int1], Img, name='Gau')
pset.addPrimitive(fe_fs.gauD, [Img, Int1, Int2, Int2], Img, name='GauD')
pset.addPrimitive(fe_fs.gab, [Img, Float1, Float2], Img, name='Gabor')
pset.addPrimitive(fe_fs.laplace, [Img], Img, name='Lap')
pset.addPrimitive(fe_fs.gaussian_Laplace1, [Img], Img, name='LoG1')
pset.addPrimitive(fe_fs.gaussian_Laplace2, [Img], Img, name='LoG2')
pset.addPrimitive(fe_fs.sobelxy, [Img], Img, name='Sobel')
pset.addPrimitive(fe_fs.sobelx, [Img], Img, name='SobelX')
pset.addPrimitive(fe_fs.sobely, [Img], Img, name='SobelY')
pset.addPrimitive(fe_fs.medianf, [Img], Img, name='Med')
pset.addPrimitive(fe_fs.meanf, [Img], Img, name='Mean')
pset.addPrimitive(fe_fs.minf, [Img], Img, name='Min')
pset.addPrimitive(fe_fs.maxf, [Img], Img, name='Max')
pset.addPrimitive(fe_fs.lbp, [Img], Img, name='LBP_F')
pset.addPrimitive(fe_fs.hog_feature, [Img], Img, name='HOG_F')
pset.addPrimitive(fe_fs.sqrt, [Img], Img, name='Sqrt')
pset.addPrimitive(fe_fs.relu, [Img], Img, name='ReLU')

# Terminals
pset.renameArguments(ARG0='Image')

pset.addEphemeralConstant('Singma', make_Sigma, Int1)
pset.addEphemeralConstant('Order', make_Order, Int2)
pset.addEphemeralConstant('Theta', make_Theta, Float1)
pset.addEphemeralConstant('Frequency', make_Frequency, Float2)
pset.addEphemeralConstant('n', make_n, Float3)
pset.addEphemeralConstant('KernelSize', make_KernelSize, Int3)

# pset.addEphemeralConstant('classificationN', lambda: random.randint(1, 4), Int4)

# Creator definitions
creator.create("FitnessMax", base.Fitness, weights=(1.0, 1.0, 1.0))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)

# Toolbox setup
toolbox = base.Toolbox()
toolbox.register("expr", gp_restrict.genHalfAndHalfMD, pset=pset, min_=initialMinDepth, max_=initialMaxDepth)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset)
toolbox.register("map", map)



def evalTrain(individual):
    try:
        func = toolbox.compile(expr=individual)
        train_tf = []
        confusion_matrices = []
        if Is_RGB:
            for i in range(len(y_train)):
                try:
                    feature = func(x_train[i][0, :, :], x_train[i][1, :, :], x_train[i][2, :, :])
                    if isinstance(feature, np.ndarray):
                        feature = feature.flatten()
                    train_tf.append(feature)
                except Exception as e:
                    # Handle any exceptions during feature extraction
                    print(f"Error evaluating individual on training data at index {i}: {e}")
                    return (0.0, 0.0, 0.0)  # Assign worst fitness if evaluation fails
        else:
            for i in range(len(y_train)):
                try:
                    feature = func(x_train[i][:, :])
                    if isinstance(feature, np.ndarray):
                        feature = feature.flatten()
                    train_tf.append(feature)
                except Exception as e:
                    # Handle any exceptions during feature extraction
                    print(f"Error evaluating individual on training data at index {i}: {e}")
                    return (0.0, 0.0, 0.0)  # Assign worst fitness if evaluation fails
        train_tf = np.array(train_tf, dtype=float)
        min_max_scaler = preprocessing.MinMaxScaler()
        train_tf = min_max_scaler.fit_transform(train_tf)
        classifier = LinearSVC(max_iter=10000)
        kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        for train_index, test_index in kfold.split(train_tf, y_train):
            X_train_sub, X_test_sub = train_tf[train_index], train_tf[test_index]
            y_train_sub, y_test_sub = y_train[train_index], y_train[test_index]
            try:
                classifier.fit(X_train_sub, y_train_sub)
                y_pred = classifier.predict(X_test_sub)
                cm = confusion_matrix(y_test_sub, y_pred, labels=unique_labels)
                confusion_matrices.append(cm)
            except Exception as e:
                print(f"Error during classifier training/prediction: {e}")
                return (0.0, 0.0, 0.0)
        if not confusion_matrices:
            return (0.0, 0.0, 0.0)
        final_cm = np.sum(confusion_matrices, axis=0)
        class_accuracies = []
        total_true = []
        total_instance_num = len(y_train)
        for label in unique_labels:
            true_positives = final_cm[label, label]
            total_true.append(true_positives)
            total_class_instances = final_cm[label, :].sum()
            if total_class_instances == 0:
                class_accuracy = 0.0
            else:
                class_accuracy = true_positives / total_class_instances
            class_accuracies.append(class_accuracy)
        accuracy_1 = sum([(1 / class_num) / sum_of_reciprocals * class_accuracie for class_num, class_accuracie in
                          zip(per_class_num, class_accuracies)])
        accuracy_2 = sum([acc / num_unique_labels for acc in class_accuracies])
        accuracy_3 = sum([acc_num for acc_num in total_true]) / total_instance_num
    except:
        accuracy_1=0
        accuracy_2=0
        accuracy_3=0
    print(accuracy_1, accuracy_2, accuracy_3)
    return accuracy_1, accuracy_2, accuracy_3


def parallel_evalTrain(individual):
    return evalTrain(individual)


# Register the evaluation function with caching
toolbox.register("evaluate", parallel_evalTrain)
toolbox.register("select", select_nsga.selNSGA2)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp_restrict.genFull, min_=0, max_=2)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)
toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=maxDepth))
toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=maxDepth))


def GPMain(pkl_file,randomSeeds):
    random.seed(randomSeeds)
    try:
        with open(pkl_file, "rb") as cp_file:
            checkpoint_data = pickle.load(cp_file)
            pop = checkpoint_data['population']
            start_gen = checkpoint_data['generation'] + 1
            print(f"Resuming from generation {start_gen}.")
    except FileNotFoundError:
        pop = toolbox.population(pop_size)
        pop = evalGP.eaSimple(pop, toolbox, cxProb, mutProb, generation, start_gen, pkl_file)
    return pop


from multiprocessing import Pool
from functools import partial


def test_individual_func(individual, toolbox, x_train, trainLabel, x_test, testLabel):
    func = toolbox.compile(expr=individual)
    print("Testing individual:", individual)
    train_tf = []
    test_tf = []
    if Is_RGB:
        for i in range(len(trainLabel)):
            train_tf.append(func(x_train[i][0, :, :], x_train[i][1, :, :], x_train[i][2, :, :]))
        train_tf = np.asarray(train_tf, dtype=float)

        for j in range(len(testLabel)):
            test_tf.append(func(x_test[j][0, :, :], x_test[j][1, :, :], x_test[j][2, :, :]))
        test_tf = np.asarray(test_tf, dtype=float)
    else:
        for i in range(len(trainLabel)):
            train_tf.append(func(x_train[i][:, :]))
        train_tf = np.asarray(train_tf, dtype=float)

        for j in range(len(testLabel)):
            test_tf.append(func(x_test[j][:, :]))
        test_tf = np.asarray(test_tf, dtype=float)

    min_max_scaler = preprocessing.MinMaxScaler()
    train_norm = min_max_scaler.fit_transform(train_tf)
    test_norm = min_max_scaler.transform(test_tf)

    classifier = LinearSVC(max_iter=10000)
    classifier.fit(train_norm, trainLabel)
    probabilities = classifier.decision_function(test_norm)
    row_min = np.min(probabilities, axis=1, keepdims=True)
    row_max = np.max(probabilities, axis=1, keepdims=True)

    row_range = row_max - row_min


    probabilities = np.divide((probabilities - row_min), row_range,
                              out=np.zeros_like(probabilities, dtype=float),
                              where=row_range != 0)
    return probabilities

def evalTest(toolbox, hof, x_train, trainLabel, x_test, testLabel):
    all_probabilities = np.zeros((len(testLabel), num_unique_labels))  # Assuming there are 10 classes

    def create_toolbox():
        local_toolbox = base.Toolbox()
        local_toolbox.register("compile", gp.compile, pset=pset)
        local_toolbox.register("mate", gp.cxOnePoint)
        local_toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

        return local_toolbox

    with Pool(processes=12) as p:
        test_func = partial(test_individual_func,toolbox = create_toolbox(), x_train=x_train, trainLabel=trainLabel, x_test=x_test,
                            testLabel=testLabel)
        results = p.map(test_func, hof)

    for probabilities in results:
        all_probabilities += probabilities

    def calculate_group_accuracy(predictions, true_labels, group_indices):
        mask = np.isin(true_labels, group_indices)
        filtered_true_labels = true_labels[mask]
        filtered_predictions = predictions[mask]
        accuracy=accuracy_score(filtered_true_labels,filtered_predictions)
        return accuracy

    final_predictions = np.argmax(all_probabilities, axis=1)
    top1_accuracy = np.mean(final_predictions == testLabel)
    top5_indices = np.argsort(all_probabilities, axis=1)[:, -5:]
    top5_matches = np.any(top5_indices == testLabel[:, None], axis=1)
    top5_accuracy = np.mean(top5_matches)
    final_report = classification_report(testLabel, final_predictions)

    accuracy_group1 = calculate_group_accuracy(final_predictions, testLabel, group1_indices)
    accuracy_group2 = calculate_group_accuracy(final_predictions, testLabel, group2_indices)
    accuracy_group3 = calculate_group_accuracy(final_predictions, testLabel, group3_indices)

    return final_predictions, final_report, top1_accuracy, top5_accuracy, accuracy_group1, accuracy_group2, accuracy_group3



if __name__ == "__main__":
    sys.stdout=Logger(opt.log_file,sys.stdout)
    start_time = time.time()
    print("Starting GP Evolution...")
    pop = GPMain(opt.pkl_file,opt.randomSeeds)
    endTime = time.process_time()
    elapsed_time = time.time() - start_time
    print(f"GP Evolution completed in {elapsed_time:.2f} seconds.")
    final_predictions, final_report, top1_accuracy, top5_accuracy, accuracy_group1, accuracy_group2, accuracy_group3 = evalTest(
        toolbox, pop, x_train, y_train, x_test, y_test
    )

    print('Final Classification Report:\n', final_report)
    print('Top1 Accuracy:', top1_accuracy)
    print('Top5 Accuracy:', top5_accuracy)
    print('accuracy_group1:', accuracy_group1)
    print('accuracy_group2:', accuracy_group2)
    print('accuracy_group3:', accuracy_group3)
    print('End')
