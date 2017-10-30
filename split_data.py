#load in tags
import pickle
from scipy import io
# import method for split train/test data set
from sklearn.model_selection import train_test_split
def split_data(data, target, portion = 0.2, shuffle = True, random_state = 100):
        #split data into train and test by 0.8/0.2
    X_train, x_test, Y_train, y_test = train_test_split(\
                        data, target, test_size = 0.2, shuffle = shuffle,\
                        random_state = 0)
    return X_train, Y_train, x_test, y_test

if __name__ == '__main__':
    #load in data
    #1gram
    dtm_1gram = pickle.load(open('dtm_1gram.pk', 'r'))
    #tags
    tags = pickle.load(open('tags.pk','r'))
    #2gram
    dtm_2gram = io.mmread('dtm_2gram.mtx')
    #split 1gram data
    x_train, train_tag, x_test, test_tag = split_data(dtm_1gram, tags)
    io.mmwrite('1gram_train.mtx', x_train)
    io.mmwrite('1gram_test.mtx', x_test)
    pickle.dump(train_tag, open('1gram_train_tags.pk','w'))
    pickle.dump(test_tag, open('1gram_test_tags.pk','w'))
    #split 2gram data
    x_train, train_tag, x_test, test_tag = split_data(dtm_2gram, tags)
    io.mmwrite('2gram_train.mtx', x_train)
    io.mmwrite('2gram_test.mtx', x_test)
    pickle.dump(train_tag, open('2gram_train_tags.pk','w'))
    pickle.dump(test_tag, open('2gram_test_tags.pk','w'))

