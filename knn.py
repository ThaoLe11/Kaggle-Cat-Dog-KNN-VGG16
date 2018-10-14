import numpy as np
    import cv2
    from matplotlib import pyplot as plt
    import sys
    import os

    lables=[]
    features=[]

    def extract_feature(img):
        img = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
        hist=cv2.calcHist([img],[0,1,2],None,[8,8,8],[0, 180, 0, 256, 0, 256])
        hist=cv2.normalize(hist)
        return hist.flatten()

    def knnOverImages(trainPath,testPath):                
        for imgPath in [ f for f in os.listdir(trainPath) if f.endswith(".jpg")]:
            img=cv2.imread(trainPath+"/"+imgPath)
            features.append(extract_feature(img))
            if imgPath.split(".")[0] == "cat":
                lables.append(1)
            else:
                lables.append(0)

        features1=np.array(features)
        lables1 = np.array(lables)
        
        # model knn
        knn=cv2.KNearest()
        knn.train(features1,lables1)
        
        # test knn               
        testData=[]
        test_labels=[]

        for imgPath in [ f for f in os.listdir(testPath) if f.endswith(".jpg")]:
            if imgPath.split(".")[0] == "cat":
                test_labels.append(0)
            else:
                test_labels.append(1)
            
            testData.append(extract_feature(cv2.imread(testPath+"/"+imgPath)))

        testData=np.array(testData)
        test_labels=np.array(test_labels)
        ret, results, neighbours, dst = knn.find_nearest(testData,9)
        
        # Now we check the accuracy of classification
        # For that, compare the result with test_labels and check which are wrong
        accuracy = len([i for i,j in zip(results.tolist(),test_labels) if i==j])*100.0/results.size
        print accuracy

    if __name__=="__main__":    
        # get path where train and test images are present
        trainPath=sys.argv[1] # path where test images are present
        testPath =sys.argv[2] # path where train images are present
        knnOverImages(trainPath,testPath)
        