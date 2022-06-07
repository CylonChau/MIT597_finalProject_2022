from random import seed
from random import randrange
from csv import reader
from math import sqrt
from math import sqrt
from math import exp
from math import pi

import copy
import datetime

from sklearn.linear_model import LogisticRegression

# 加载CSV
def load_csv(filename):
    dataset = list()
    n = 0
    with open(filename, 'r') as file:
        csv_reader = reader(file)
        for row in csv_reader:
            del(row[0])
            if not row:
                continue
            if n == 0:
                n += 1
                continue
            else:
                dataset.append(row)
                n += 1
    return dataset

# 转换所有的值为float方便运算
def str_column_to_float(dataset, column):
    for row in dataset:
        row[column] = float(row[column])

# 转换所有的类型为int
def str_column_to_int(dataset, column):
    class_values = [row[column] for row in dataset]
    unique = set(class_values)
    lookup = dict()
    for i, value in enumerate(unique):
        lookup[value] = i
    for row in dataset:
        row[column] = lookup[row[column]]
    return lookup

class Algorithm:
    # k-folds CV函数进行划分
    def cross_validation_split(self, dataset, n_folds):
        dataset_split = list()
        dataset_copy = list(dataset)
        # 平均分成n_folds折数
        fold_size = int(len(dataset) / n_folds)
        for _ in range(n_folds):
            fold = list()
            while len(fold) < fold_size:
                index = randrange(len(dataset_copy))
                fold.append(dataset_copy.pop(index))
            dataset_split.append(fold)
        return dataset_split

    # 计算精确度
    def accuracy_metric(self, actual, predicted):
        correct = 0
        for i in range(len(actual)):
            if actual[i] == predicted[i]:
                correct += 1
        return correct / float(len(actual)) * 100.0

    # 评估算法
    def evaluate_algorithm(self, dataset, algorithm, n_folds, *args):
        """
        评估算法，计算算法的精确度
        :param dataset: list, 数据集
        :param algorithm: function, 算法名
        :param n_folds: int，折数
        :param args: 用于algorithm的参数
        :return: None
        """
        folds = self.cross_validation_split(dataset, n_folds) # 分成5折
        scores = list()
        count = 1
        for fold in folds:
            train_set = list(folds)
            train_set.remove(fold) # 训练集不包含本身

            train_set = sum(train_set, [])
            test_set = list() # 测试集
            for row in fold:
                row_copy = list(row)
                test_set.append(row_copy)
                row_copy[-1] = None
            print("Start the %dst predicted." % count)
            predicted = algorithm(train_set, test_set, *args)
            actual = [row[-1] for row in fold]
            accuracy = self.accuracy_metric(actual, predicted)
            scores.append(accuracy)
            count += 1
        return scores


class KNN(Algorithm):

    # 欧几里得距离，计算两个向量间距离的算法
    def euclidean_distance(self, row1, row2):
        distance = 0.0
        for i in range(len(row1)-1):
            distance += (row1[i] - row2[i])**2
        return sqrt(distance)

    # 确定最邻近的邻居
    def get_neighbors(self, train, test_row, num_neighbors):
        """
        计算训练集train中所有元素到test_row的距离
        :param train: list, 数据集，可以是训练集
        :param test_row: list, 新的实例，也就是K
        :param num_neighbors:int，需要多少个邻居
        :return: None
        """
        distances = list()
        for train_row in train:
            dist = self.euclidean_distance(test_row, train_row)
            distances.append((train_row, dist))
        distances.sort(key=lambda tup: tup[1])
        neighbors = list()
        for i in range(num_neighbors):
            neighbors.append(distances[i][0])
        return neighbors

    # 与临近值进行比较并预测
    def predict_classification(self, train, test_row, num_neighbors):
        """
        计算训练集train中所有元素到test_row的距离
        :param train: list, 数据集，可以是训练集
        :param test_row: list, 新的实例，也就是K
        :param num_neighbors:int，需要多少个邻居
        :return: None
        """
        neighbors = self.get_neighbors(train, test_row, num_neighbors)
        output_values = [row[-1] for row in neighbors]
        prediction = max(set(output_values), key=output_values.count)
        return prediction
    
    # kNN Algorithm
    def k_nearest_neighbors(self, train, test, num_neighbors):
        predictions = list()
        for row in test:
            output = self.predict_classification(train, row, num_neighbors)
            predictions.append(output)
        return(predictions)

class DecisionTree(Algorithm):
    # 根据基尼指数划分value是应该在树的哪边？
    def test_split(self, index, value, dataset):
        left, right = list(), list()
        for row in dataset:
            if row[index] < value:
                left.append(row)
            else:
                right.append(row)
        return left, right
    
    # 基尼指数打分
    def gini_index(self, groups, classes):
        # 计算数据集中的多组数据的总个数
        n_instances = float(sum([len(group) for group in groups]))
        # 计算每组中的最优基尼指数
        gini = 0.0
        for group in groups:
            size = float(len(group))
            if size == 0:
                continue
            score = 0.0
            # 总基尼指数
            for class_val in classes:
                # 拿出数据集中每行的类型，拆开是为了更好的了解结构

                # 计算的是当前的分类在总数据集中占比
                p = [row[-1] for row in group]
                p1 = p.count(class_val) / size
                score += p1 * p1
            # 计算总的基尼指数，并根据相应大小增加权重。权重：当前分组占总数据集中的数量
            gini += (1.0 - score) * (size / n_instances)
        return gini

    # 从数据集中获得基尼指数最佳的值
    def get_split(self, dataset):
        class_values = list(set(row[-1] for row in dataset))
        b_index, b_value, b_score, b_groups = 999, 999, 999, None
        for index in range(len(dataset[0])-1):
            for row in dataset:
                groups = self.test_split(index, row[index], dataset)
                gini = self.gini_index(groups, class_values)
                if gini < b_score:
                    b_index, b_value, b_score, b_groups = index, row[index], gini, groups
        return {'index':b_index, 'value':b_value, 'groups':b_groups}

    # 创建终端节点
    def to_terminal(self, group):
        outcomes = [row[-1] for row in group]
        return max(set(outcomes), key=outcomes.count)

        # 创建子节点，为终端节点或子节点
    def split(self, node, max_depth, min_size, depth):
        """
        :param node: {},分割好的的{'index':b_index, 'value':b_value, 'groups':b_groups}
        :param max_depth: int, 最大深度
        :param min_size:int，最小模式数
        :param depth:int， 当前深度
        :return: None
        """
        left, right = node['groups']
        del(node['groups'])
        # check for a no split
        if not left or not right:
            node['left'] = node['right'] = self.to_terminal(left + right)
            return
        # check for max depth
        if depth >= max_depth:
            node['left'], node['right'] = self.to_terminal(left), self.to_terminal(right)
            return
        # process left child
        if len(left) <= min_size:
            node['left'] = self.to_terminal(left)
        else:
            node['left'] = self.get_split(left)
            self.split(node['left'], max_depth, min_size, depth+1)
        # process right child
        if len(right) <= min_size:
            node['right'] = self.to_terminal(right)
        else:
            node['right'] = self.get_split(right)
            self.split(node['right'], max_depth, min_size, depth+1)
 
    # 构建树
    def build_tree(self, train, max_depth, min_size):
        """
        :param train: list, 数据集，可以是训练集
        :param max_depth: int, 最大深度
        :param min_size:int，最小模式数
        :ret
        """
        root = self.get_split(train)
        self.split(root, max_depth, min_size, 1)
        return root
    
    # 预测，预测方式为当前基尼指数与最优基尼指数相比较，然后放入树两侧
    def predict(self, node, row):
        """
        :param node: {} 叶子值
        :param row: {}, 需要预测值
        :ret
        """
        if row[node['index']] < node['value']:
            if isinstance(node['left'], dict):
                return self.predict(node['left'], row)
            else:
                return node['left']
        else:
            if isinstance(node['right'], dict):
                return self.predict(node['right'], row)
            else:
                return node['right']

    def decision_tree(self, train, test, max_depth, min_size):
        tree = self.build_tree(train, max_depth, min_size)
        predictions = list()
        for row in test:
            prediction = self.predict(tree, row)
            predictions.append(prediction)
        return(predictions)

class NaiveBayes(Algorithm):
    # 按照分类拆分
    def separate_by_class(self, dataset):
        """
        :param dataset:list, 按分类好的列表
        :return: dict, 每个分类的每列（属性）的平均值，标准差，个数
        """
        separated = dict()
        for i in range(len(dataset)):
            vector = dataset[i]
            class_value = vector[-1]
            if (class_value not in separated):
                separated[class_value] = list()
            separated[class_value].append(vector)
        return separated
    
    # 计算这一系列的平均值
    def mean(self, numbers):
        # if sum(numbers) == 0:
        #     print(sum(numbers))
        #     print(len(numbers))
        return sum(numbers)/float(len(numbers))

    # 计算一系列数字的标准差
    def stdev(self, numbers):
        avg = self.mean(numbers)
        variance = sum([(x-avg)**2 for x in numbers]) / float(len(numbers)-1)
        return sqrt(variance)

    # 计算数据集中每列的平均值 标准差 长度
    def summarize_dataset(self, dataset):
        summaries = [(self.mean(column), self.stdev(column), len(column)) for column in zip(*dataset)]
        del(summaries[-1])
        return summaries

    # 按照分类划分数据集
    def summarize_by_class(self, dataset):
        separated = self.separate_by_class(dataset)
        summaries = dict()
        for class_value, rows in separated.items():
            summaries[class_value] = self.summarize_dataset(rows)

        return summaries

    # 计算x的高斯概率
    def calculate_probability(self, x, mean, stdev):
        """
        :param x:float, 计算这个值的高斯概率
        :param mean:float，x的平均值
        :param stdev:float，x的标准差
        :return: None
        """
        # print("x--%s" % x)
        # print("mean--%s" % mean)
        # print("stdev--%s" % stdev)
        if mean == 0 and stdev == 0:
            return 0
        exponent = exp(-((x-mean)**2 / (2 * stdev**2 )))
        return (1 / (sqrt(2 * pi) * stdev)) * exponent
    # 计算每行的概率
    def converge_probabilities(self, summaries, row):
        # 计算所有分类的个数
        total_rows = sum([summaries[label][0][2] for label in summaries])
        probabilities = dict()
        for class_value, class_summaries in summaries.items():
            # 计算分类的概率，如这个分类在总分类里概率多少
            # 公式中的P(class)
            probabilities[class_value] = summaries[class_value][0][2]/float(total_rows)
            # 通过公式  P(X1|class=0) * P(X2|class=0) * P(class=0) 计算高斯概率
            for i in range(len(class_summaries)):
                mean, stdev, _ = class_summaries[i]
                probabilities[class_value] *= self.calculate_probability(row[i], mean, stdev)
        return probabilities

    # 通过计算出来的值，预测该花属于哪个品种，取高斯概率最大的值
    def predict(self, summaries, row):
        probabilities = self.converge_probabilities(summaries, row)
        best_label, best_prob = None, -1
        for class_value, probability in probabilities.items():
            if best_label is None or probability > best_prob:
                best_prob = probability
                best_label = class_value
        return best_label
    
    # Naive Bayes Algorithm
    def naive_bayes(self, train, test):
        # 训练数据按照类分类排序
        summarize = self.summarize_by_class(train)
        predictions = list()
        for row in test:
            output = self.predict(summarize, row)
            predictions.append(output)
        return(predictions)

class LogisticRegression(Algorithm):
    # 预测函数
    def predict(self, row, coefficients):
        p = coefficients[0]
        for i in range(len(row)-1):
            p += coefficients[i + 1] * row[i]
        return 1.0 / (1.0 + exp(-p))
    # 系数生成
    def coefficients_sgd(self, train, l_rate, n_epoch):
        coef = [0.0 for i in range(len(train[0]))] # 初始一个系数，第一次为都为0
        for epoch in range(n_epoch):
            sum_error = 0
            for row in train:
                p = self.predict(row, coef)
                # 错误为预期值与实际值直接差异
                error = row[-1] - p
                sum_error += error**2
                # 截距没有输入变量x，这里为row[0]
                coef[0] = coef[0] + l_rate * error * p * (1.0 - p)
                for i in range(len(row)-1):
                    # 其他系数更新
                    coef[i + 1] = coef[i + 1] + l_rate * error * p * (1.0 - p) * row[i]
            # print('>epoch=%d, lrate=%.3f, error=%.3f' % (epoch, l_rate, sum_error))
        return coef
    # 归一化
    def normalize_dataset(self, dataset, minmax):
        for row in dataset:
            for i in range(len(row)):
                row[i] = (row[i] - minmax[i][0]) / (minmax[i][1] - minmax[i][0])
    # Find the min and max values for each column
    def dataset_minmax(self, dataset):
        minmax = list()
        for i in range(len(dataset[0])):
            col_values = [row[i] for row in dataset]
            value_min = min(col_values)
            value_max = max(col_values)
            minmax.append([value_min, value_max])
        return minmax

    # 随机梯度下降的逻辑回归算法
    def logistic_regression(self, train, test, l_rate, n_epoch):
        predictions = list()
        coef = self.coefficients_sgd(train, l_rate, n_epoch)
        for row in test:
            p = self.predict(row, coef)
            p = round(p)
            predictions.append(p)
        return(predictions)

def default_case(s):
    print("")
    print("Invalid selected, Please select algorithms: \n 1. K-NN: \n 2. Naive Bayes:\n 3. Decision Tree:\n 4. All algorithms")

def NewKNN():
    knn = KNN()
    n_folds = 5 # 5折
    num_neighbors = 5 #取5个邻居
    startTime = datetime.datetime.now()
    print("Start time: %s" % startTime)
    knn_scores = knn.evaluate_algorithm(dataset, knn.k_nearest_neighbors, n_folds, num_neighbors)
    endTime = datetime.datetime.now()
    print("end time: %s" % endTime)
    duringTime = endTime - startTime
    print("time comsumption: %s" % duringTime)
    print('Scores: %s' % knn_scores)
    print('Mean Accuracy: %.3f%%' % (sum(knn_scores)/float(len(knn_scores))))

def NewDecisionTree():
    DT = DecisionTree()
    n_folds = 5 # 5折
    max_depth = 5
    min_size = 10
    startTime = datetime.datetime.now()
    print("Start time: %s" % startTime)
    dt_scores = DT.evaluate_algorithm(dataset, DT.decision_tree, n_folds, max_depth, min_size)
    endTime = datetime.datetime.now()
    print("end time: %s" % endTime)
    duringTime = endTime - startTime
    print("time comsumption: %s" % duringTime)
    print('Scores: %s' % dt_scores)
    print('Mean Accuracy: %.3f%%' % (sum(dt_scores)/float(len(dt_scores))))

def NewNaiveBayse():
    NB = NaiveBayes()
    n_folds = 5 # 5折

    startTime = datetime.datetime.now()
    print("Start time: %s" % startTime)
    nb_scores = NB.evaluate_algorithm(dataset, NB.naive_bayes, n_folds)
    endTime = datetime.datetime.now()
    print("end time: %s" % endTime)
    duringTime = endTime - startTime
    print("time comsumption: %s" % duringTime)
    print('Scores: %s' % nb_scores)
    print('Mean Accuracy: %.3f%%' % (sum(nb_scores)/float(len(nb_scores))))

def NewLogisticRegression():
    LR = LogisticRegression()
    n_folds = 5 # 5折
    learnning_rate = 0.5 # 学习率
    n_epoch = 100

    copy_dataset = copy.deepcopy(dataset)

    startTime = datetime.datetime.now()
    print("Start time: %s" % startTime)
    # 归一化
    minmax = LR.dataset_minmax(copy_dataset)
    LR.normalize_dataset(copy_dataset, minmax)
    lr_scores = LR.evaluate_algorithm(copy_dataset, LR.logistic_regression, n_folds, learnning_rate, n_epoch)
    endTime = datetime.datetime.now()
    print("end time: %s" % endTime)
    duringTime = endTime - startTime
    print("time comsumption: %s" % duringTime)
    print('Scores: %s' % lr_scores)
    print('Mean Accuracy: %.3f%%' % (sum(lr_scores)/float(len(lr_scores))))


def NewAll():
    NewKNN()
    NewNaiveBayse()
    NewLogisticRegression()
    NewDecisionTree()

if __name__ == "__main__":
    
    print("This Scripts will evaluation machine learning algorithms, \n   Please waitting process dataset.\n")
    seed(1)
    filename = 'emails.csv'
    # filename = 'pima-indians-diabetes.csv'
    dataset = load_csv(filename)
    # dataset = dataset[:100]

    for i in range(len(dataset[0])-1):
        str_column_to_float(dataset, i)

    # 转换类型为int
    str_column_to_int(dataset, len(dataset[0])-1)
    
    print("Please select algorithm: \n 1. K-NN \n 2. Naive Bayes\n 3. Decision Tree\n 4. Logistic Regression\n 5. All algorithms")

    switch = {
        1: NewKNN,
        2: NewNaiveBayse,
        3: NewDecisionTree,
        4: NewLogisticRegression,
        5: NewAll,
    }
    m = 1
    while (m == 1):
        option = input("Please select algorithms (tips: input int 1-4 to select，other key exit.): ")
        option = int(option) if option.isdigit() else 0
        if(option >=1 and option <=4):
            switch.get(option, default_case)()
        else:
            print("exit.")
            m = 0