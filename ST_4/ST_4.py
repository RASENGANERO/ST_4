import sys
import cv2
import numpy as np
import random
import math
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (QWidget, QApplication, QLabel, QVBoxLayout, QHBoxLayout, QPushButton, 
                             QFileDialog,QComboBox,QMessageBox,QTextBrowser)
from PyQt5.QtCore import Qt
class Example(QWidget):
    def __init__(self):
        super().__init__()
        self.image = None
        self.textbrowser = QTextBrowser()
        self.initUI()

    def initUI(self):
        

        self.btn_open = QPushButton('Изображения folder1')
        self.btn_open.clicked.connect(self.openImages1)

        self.btn_open1 = QPushButton('Изображения folder2')
        self.btn_open1.clicked.connect(self.openImages2)


        self.btn_ = QPushButton('Обучить и создать матрицы')
        self.btn_.clicked.connect(self.matrix_and_train)


        self.btn1 = QPushButton('Детектировать тестовые изображения')
        self.btn1.clicked.connect(self.dec)

        self.btn2 = QPushButton('Сформировать лес и натренировать полученную модель')
        self.btn2.clicked.connect(self.form_les_and_train)

        self.btn_.setVisible(False)
        self.btn1.setVisible(False)
        self.btn2.setVisible(False)


        top_bar = QHBoxLayout()
        top_bar.addWidget(self.btn_open)
        top_bar.addWidget(self.btn_open1)
        top_bar.addWidget(self.btn_)
        top_bar.addWidget(self.btn1)
        top_bar.addWidget(self.btn2)
        root = QVBoxLayout(self)
        root.addLayout(top_bar)
        root.addWidget(self.textbrowser)
        
        self.spisok=list()
        self.spisok2=list()
        self.spisok3=list()


        self.train=list()
        self.matrix=list()
        self.resize(540, 574)
        self.setWindowTitle('ST_4')
        self.show()

    def openImages1(self):
        filenames1 = QFileDialog.getOpenFileNames(None, 'Открыть изображения', '.', 'Image Files (*.png *.jpg *.jpeg *.bmp)')
        lk=filenames1[0]
        self.erroropened(lk,"1")
        self.mass(lk,1)
        lk.clear()


    def openImages2(self):
        filenames2 = QFileDialog.getOpenFileNames(None, 'Открыть изображения', '.', 'Image Files (*.png *.jpg *.jpeg *.bmp)')
        lk=filenames2[0]
        self.erroropened(lk,"2")
        self.mass(lk,2)
        lk.clear()
        
    def erroropened(self,s,s1):
        if len(s)!=0:
            q=QMessageBox.information(self,"Информация","Изображения из "+s1+" папки получены!")
        else:
            q=QMessageBox.information(self,"Информация","Вы не выбрали изображения!")


    def mass(self,p,s1):
        if s1==1:
            self.spisok3.clear()
            for v in range(len(p)):
                self.spisok.append(str(p[v]))
                self.spisok3.append(str(p[v])+"SKT")
            self.appendos("Тестовый набор картинок",self.spisok)
        if s1==2:
            for v in range(len(p)):
                self.spisok2.append(str(p[v]))
                self.spisok3.append(str(p[v])+"NO")
            self.spisok3=list(set(self.spisok3))
            self.appendos("Набор картинок для тренировки",self.spisok2)
            self.btn_open.setVisible(False)
            self.btn_open1.setVisible(False)
            self.btn_.setVisible(True)
        q=QMessageBox.information(self,"Информация","Количество изображений тестовой категории: "+str(len(self.spisok))+"\nКоличество изображений основной категории: "+str(len(self.spisok2))+"\nОбщее количество изображений: "+str(int(len(self.spisok)+int(len(self.spisok2)))))
    



    def matrix_and_train(self):
        for v in range(len(self.spisok3)):
            s=str(self.spisok3[v])
            if s.endswith("SKT"):
                self.train.append(round(float(0.5),1))
                self.matrix.append(round(float(1.0),1))
            if s.endswith("NO"):
                q=round(float(random.uniform(1.0,3.0)),1)                
                self.train.append(q)
                self.matrix.append(round(float(-1.0),1))
                
        self.appendos("Ваши тренировочные данные",self.train)
        self.appendos("Ваша матрица",self.matrix)
        train=np.array([self.train],dtype=int)
        labels = np.array(self.matrix,dtype=int)
        self.svm = cv2.ml.SVM_create()
        self.svm.train(train, cv2.ml.COL_SAMPLE, labels)
        self.svm.save("1.yml")
        self.textbrowser.append("Модель сохранена!")
        close = QMessageBox.question(self,"Поздравляем!","Ваша модель натренирована!",QMessageBox.Yes | QMessageBox.No)
        if close == QMessageBox.Yes:
            pass
        self.btn_.setVisible(False)
        self.btn1.setVisible(True)

    def appendos(self,s1,s2):
        self.textbrowser.append(s1)
        for v in range(len(s2)):
            self.textbrowser.append(str(s2[v]))

    def dec(self):
        self.tr1=list()
        self.tr2=list()
        self.model = RandomForestClassifier(n_estimators=100,bootstrap = True,max_features = 'sqrt')
        self.train_got(self.tr1,self.spisok)
        self.train_got(self.tr2,self.spisok2)
        self.btn1.setVisible(False)
        self.btn2.setVisible(True)

    def train_got(self,r,s):
        for v in range(len(s)):
            lkst=list()
            img = cv2.imread(str(s[v]))
            gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            sift = cv2.SIFT_create()
            kp = sift.detect(gray,None)
            self.textbrowser.append("Натренирована модель с данными изображения "+str(v+1))
            for keyPoint in kp:
                lkst.append(keyPoint.pt[0])
                lkst.append(keyPoint.pt[1])
            train=np.array([lkst],dtype=int)
            labels = train
            self.model.fit(train, labels)
            self.model.predict(labels)
            predictions=self.model.predict(labels)
            for p in range(len(predictions)):
                ret=predictions[p]
                r.append(ret)







    def form_les_and_train(self):
        
        self.spisok3.clear()
        for v in range(len(self.tr2)):
            q1=self.tr2[v]
            q2=self.tr2[v]
            lkst1=list()
            lkst2=list()
            for p in range(len(q1)):
                lkst1.append(q1[p])
            for p in range(len(q2)):
                lkst2.append(q2[p])
            if len(lkst1)==len(lkst2):
                self.spisok3.append(self.spisok2[v])
                print("Длина исходного изображения "+str(len(lkst1))+" Длина тестового изображения"+str(len(lkst2)))
            else:
                pass
        for v in range(len(self.spisok3)):
            img = cv2.imread(str(self.spisok3[v]))
            cv2.imshow("Original image", img)
            cv2.waitKey(0)

        #self.tr2.clear()
        #self.spisok3.clear()
        #for v in range(len(self.spisok2)):
        #    lkst=list()
        #    img = cv2.imread(str(self.spisok2[v]))
          #  gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
          #  sift = cv2.SIFT_create()
          #  kp = sift.detect(gray,None)
          #  self.textbrowser.append("Натренирована модель с данными изображения "+str(v+1))
          #  for keyPoint in kp:
         #       lkst.append(keyPoint.pt[0])
         #       lkst.append(keyPoint.pt[1])
         #   train=np.array([lkst],dtype=int)
         #   for p in range(len(train)):
         #       ret=train[p]
         #       self.tr2.append(ret)
            
        #print(len(self.tr1))
        #print(len(self.tr2))
        #for v in range(len(self.tr2)):
           # q1=self.tr2[v]
           # q2=self.tr1[v]
           # lkst1=list()
           # lkst2=list()
          #  for p in range(len(q1)):
          #      lkst1.append(q1[p])
          #  for p in range(len(q2)):
          #      lkst2.append(q2[p])

            #print(lkst1)
            #print(lkst2)
         #   if len(lkst1)==len(lkst2):
         #       print("YES")
         #   print(str(sum(lkst1))+" "+str(sum(lkst2)))



   #     labels = np.array([self.spisok3],dtype=int)
   #     self.model.predict(labels)
  #      self.textbrowser.append("Натренирована модель с данными изображения ")    
  #      self.textbrowser.append("Предсказаны изображения на тестовой выборке! ")
  #      predictions = self.model.predict(labels)
  #      print(predictions)
  #      for v in range(len(predictions)):
  #          s=predictions[v]
  #          for q in range(len(s)):
  #              print(s[q])

    


    def closeEvent(self, event):
        close = QMessageBox.question(self,"Выход","Вы хотите завершить работу?",QMessageBox.Ok | QMessageBox.No)
        if close == QMessageBox.Ok:
            event.accept()
        else:
            event.ignore()



if __name__ == '__main__':
    app = QApplication(sys.argv)
    win = Example()
    sys.exit(app.exec_())