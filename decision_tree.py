import sklearn
from sklearn import tree
#data is obtained by me.(using cv2 to get pixel coordinate of ant)
#in features 1st  is the coordinate of ant and 2nd  is time.
#label shows whether it is at that position on that time or not.
#then we have trained our model
features = [[179, 0], [131, 2], [126, 3], [157, 4], [131, 5], [131, 6], [179, 7], [125, 8], [129, 9], [177, 10], [150, 13], [111, 14], [123, 15]]
label = [1,1,1,1,1,1,1,1,1,0,0,1,1]
for i in range(100):

    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(features,label)
    print(clf.predict([[118, 16], [115, 18], [144, 20],[231,21],[150,22],[190,23]]))


