from PyQt5.QtWidgets import QMainWindow, QApplication, QFileDialog, QMessageBox, QColorDialog
from PyQt5.QtGui import QPixmap, QImage
from PyQt5 import uic, QtGui, QtCore
import sys
import math
import random
from collections import deque
import numpy as np, cv2

class MainWindow(QMainWindow, uic.loadUiType('main.ui')[0]):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.setWindowTitle('Final Project')

        self.fileName = None
        self.pixmap = None
        self.image = None
        self.orgImage = None
        self.tmpImage = None
        self.undoQueue = deque()
        self.redoQueue = deque()
        self.x1 = 0
        self.y1 = 0
        self.x2 = 0
        self.y2 = 0
        self.button = -1
        self.color = None
        self.mask = None

        self.imageLabel.mousePressEvent = self.mousePressed
        self.imageLabel.mouseMoveEvent = self.mouseMoved
        self.imageLabel.mouseReleaseEvent = self.mouseReleased
        self.actionOpen.triggered.connect(self.fileOpen)
        self.actionSave.triggered.connect(self.fileSave)
        self.paletteButton.clicked.connect(self.paletteButtonClicked)
        self.randomButton.clicked.connect(self.randomButtonClicked)
        self.removeButton.clicked.connect(self.removeButtonClicked)
        self.mosaicButton.clicked.connect(self.mosaicButtonClicked)
        self.correctionButton.clicked.connect(self.correctionButtonClicked)
        self.edgeButton.clicked.connect(self.edgeButtonClicked)
        self.webtoonButton.clicked.connect(self.webtoonButtonClicked)
        self.sketchButton.clicked.connect(self.sketchButtonClicked)
        self.undoButton.clicked.connect(self.undoButtonClicked)
        self.redoButton.clicked.connect(self.redoButtonClicked)
        self.returnButton.clicked.connect(self.returnButtonClicked)
        self.rTrackbar.valueChanged.connect(self.rValueChanged)
        self.gTrackbar.valueChanged.connect(self.gValueChanged)
        self.bTrackbar.valueChanged.connect(self.bValueChanged)
        self.rTrackbar.sliderReleased.connect(self.trackbarReleased)
        self.gTrackbar.sliderReleased.connect(self.trackbarReleased)
        self.bTrackbar.sliderReleased.connect(self.trackbarReleased)

        self.initUi()

    def initUi(self):
        self.undoQueue.clear()
        self.redoQueue.clear()
        self.x1 = 0
        self.y1 = 0
        self.x2 = 0
        self.y2 = 0
        self.button = -1
        self.undoButton.setEnabled(False)
        self.redoButton.setEnabled(False)
        self.returnButton.setEnabled(False)
        self.rTrackbar.setValue(0)
        self.gTrackbar.setValue(0)
        self.bTrackbar.setValue(0)

    def fileOpen(self):
        self.fileName = QFileDialog.getOpenFileName(self, 'Open File', '', '모든 파일(*);; PNG(*.png);; JPEG(*.jpg;*jpeg;*.jpe;*.jfif)')[0]
        if self.fileName:
            self.orgImage = cv2.cvtColor(cv2.imread(self.fileName, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
            self.image = self.orgImage.copy()
            self.tmpImage = self.orgImage.copy()
            self.updateImageLabel(self.image)
            self.initUi()
        else:
            QMessageBox.about(self, 'Warning', '파일을 선택하지 않았습니다.')

    def fileSave(self):
        if self.pixmap:
            fileName = QFileDialog.getSaveFileName(self, 'Save File', '', '모든 파일(*);; PNG(*.png);; JPEG(*.jpg;*jpeg;*.jpe;*.jfif)')[0]
            if fileName:
                self.pixmap.save(fileName)
        else:
            QMessageBox.about(self, 'Warning', '저장할 파일이 없습니다.')

    def mousePressed(self, event):
        if self.pixmap:
            self.x1 = event.x()
            self.y1 = event.y()
            print('x1: {0}, y1: {1}'.format(self.x1, self.y1))

            if self.button == 0:
                self.runPaletteButton()

    def mouseMoved(self, event):
        if self.pixmap:
            if self.button == 3:
                self.imageLabel.setCursor(QtGui.QCursor(QtCore.Qt.ClosedHandCursor))
                x = event.x()
                y = event.y()
                size = (abs(self.x1 - x) // 2, abs(self.y1 - y) // 2)
                center = (min(self.x1, x) + size[0], min(self.y1, y) + size[1])
                self.tmpImage = self.image.copy()
                cv2.ellipse(self.tmpImage, center, size, 0, 0, 360, (255, 0, 0), 1)
                self.updateImageLabel(self.tmpImage)

    def mouseReleased(self, event):
        if self.pixmap:
            self.x2 = event.x()
            self.y2 = event.y()
            print('x2: {0}, y2: {1}'.format(self.x2, self.y2))

            if self.button == -1:
                self.liquify()
                self.updateQueue()
                self.updateImageLabel(self.image)
            elif self.button == 0:
                self.button = -1
                self.imageLabel.setCursor(QtGui.QCursor(QtCore.Qt.ArrowCursor))
            elif self.button == 3:
                size = (abs(self.x1 - self.x2) // 2, abs(self.y1 - self.y2) // 2)
                center = (min(self.x1, self.x2) + size[0], min(self.y1, self.y2) + size[1])
                self.mask = np.zeros_like(self.image)
                cv2.ellipse(self.mask, center, size, 0, 0, 360, (255, 255, 255), -1)
                self.runMosaicButton()
                self.button = -1
                self.imageLabel.setCursor(QtGui.QCursor(QtCore.Qt.ArrowCursor))

    def liquify(self):
        half = 30
        x, y, w, h = self.x1-half, self.y1-half, half*2, half*2
        if abs(self.x1 - self.x2) > 5 or abs(self.y1 - self.y2) > 5:
            self.tmpImage = self.image.copy()
            roi = self.tmpImage[y:y+h, x:x+w].copy()
            dst = roi.copy()

            offset_cx1, offset_cy1 = self.x1-x, self.y1-y
            offset_cx2, offset_cy2 = self.x2-x, self.y2-y

            tri1 = [[[0, 0], [w, 0], [offset_cx1, offset_cy1]],
                    [[0, 0], [0, h], [offset_cx1, offset_cy1]],
                    [[w, 0], [offset_cx1, offset_cy1], [w, h]],
                    [[0, h], [offset_cx1, offset_cy1], [w, h]]]
            tri2 = [[[0, 0], [w, 0], [offset_cx2, offset_cy2]],
                    [[0, 0], [0, h], [offset_cx2, offset_cy2]],
                    [[w, 0], [offset_cx2, offset_cy2], [w, h]],
                    [[0, h], [offset_cx2, offset_cy2], [w, h]]]

            for i in range(4):
                matrix = cv2.getAffineTransform(np.float32(tri1[i]), np.float32(tri2[i]))
                warped = cv2.warpAffine(roi, matrix, (w, h), None, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
                mask = np.zeros((h, w), np.uint8)
                cv2.fillConvexPoly(mask, np.int32(tri2[i]), (255, 255, 255))
                warped = cv2.bitwise_and(warped, warped, mask=mask)
                dst = cv2.bitwise_and(dst, dst, mask=cv2.bitwise_not(mask))
                dst = dst + warped

            self.tmpImage[y:y+h, x:x+w] = dst

    def paletteButtonClicked(self):
        if self.pixmap:
            col = QColorDialog.getColor()
            if col.isValid():
                self.color = list(col.getRgb())[:3]
                self.button = 0
                self.imageLabel.setCursor(QtGui.QCursor(QtCore.Qt.CrossCursor))

    def runPaletteButton(self):
        dx = [-1, 0, 1, 1, 1, 0, -1, -1]
        dy = [-1, -1, -1, 0, 1, 1, 1, 0]
        visited = [[False] * self.image.shape[1] for _ in range(self.image.shape[0])]

        def BFS():
            queue = deque()
            src = self.image.copy()
            src[self.y1][self.x1] = self.color
            visited[self.y1][self.x1] = True
            queue.append((self.x1, self.y1))
            while queue:
                x, y = queue.popleft()
                for i in range(8):
                    nx = x + dx[i]
                    ny = y + dy[i]
                    if 0 <= nx < self.image.shape[1] and 0 <= ny < self.image.shape[0]:
                        if not visited[ny][nx]:
                            r, g, b = self.image[y][x]
                            if r-2 <= src[ny][nx][0] <= r+2 and g-2 <= src[ny][nx][1] <= g+2 and b-2 <= src[ny][nx][2] <= b+2:
                                src[ny][nx] = self.color
                                visited[ny][nx] = True
                                queue.append((nx, ny))
            return src

        self.tmpImage = BFS()
        self.updateQueue()
        self.updateImageLabel(self.image)

    def randomButtonClicked(self):
        if self.pixmap:
            self.button = -1
            self.imageLabel.setCursor(QtGui.QCursor(QtCore.Qt.ArrowCursor))

            def find_nearest(array, value):
                array = array[np.where((np.abs(array[:,0] - value[0]) <= 10) & (np.abs(array[:,1] - value[1]) <= 50))]
                idx = np.abs(array - value).argmin()
                return array[idx // 3]

            randomPalette = random.sample(list((np.random.rand(300, 3) * 256).astype(np.uint8)), 254)
            randomPalette = np.append(randomPalette, np.array([[0, 0, 0]]), axis=0)
            randomPalette = np.append(randomPalette, np.array([[255, 255, 255]]), axis=0)

            self.tmpImage = cv2.cvtColor(self.orgImage.copy(), cv2.COLOR_RGB2HSV)
            for i in range(self.tmpImage.shape[0]):
                print('-----------------------')
                print('before : ', end='')
                print(self.tmpImage[i][0])
                for j in range(self.tmpImage.shape[1]):
                    self.tmpImage[i][j] = find_nearest(randomPalette, self.tmpImage[i][j])
                print('after  : ', end='')
                print(self.tmpImage[i][0])
                print('-----------------------')
            self.tmpImage = cv2.cvtColor(self.tmpImage, cv2.COLOR_HSV2RGB)

            # orgColorThief = ColorThief(self.fileName)
            # orgPalette = orgColorThief.get_palette(quality=1)
            # plt.subplot(2, 1, 1)
            # plt.imshow([[orgPalette[i] for i in range(len(orgPalette))]])
            # fileName = './tmp.jpg'
            # cv2.imwrite(fileName, cv2.cvtColor(self.tmpImage, cv2.COLOR_RGB2BGR))
            # tmpColorThief = ColorThief(fileName)
            # tmpPalette = tmpColorThief.get_palette(quality=1)
            # plt.subplot(2, 1, 2)
            # plt.imshow([[tmpPalette[i] for i in range(len(tmpPalette))]])
            # plt.show()

            self.updateQueue()
            self.updateImageLabel(self.image)

    def removeButtonClicked(self):
        if self.pixmap:
            self.button = -1
            self.imageLabel.setCursor(QtGui.QCursor(QtCore.Qt.ArrowCursor))

            def inv_relu(input):
                return 0 if input > 0 else input

            def water_filling(input):
                input.astype(np.float32)
                h = cv2.resize(input, (0, 0), fx=0.2, fy=0.2, interpolation=cv2.INTER_LINEAR)

                height = h.shape[0]
                width = h.shape[1]
                w = np.zeros_like(h, np.float32)
                G = np.zeros_like(h, np.float32)

                for t in range(2500):
                    G = w + h
                    _, G_max, _, _ = cv2.minMaxLoc(G, None)
                    for y in range(1, height-2):
                        for x in range(1, width-2):
                            tmp = (0.2 * (inv_relu(-G[y][x] + G[y+1][x])
                                        + inv_relu(-G[y][x] + G[y-1][x])
                                        + inv_relu(-G[y][x] + G[y][x+1])
                                        + inv_relu(-G[y][x] + G[y][x-1]))) + (math.exp(-t) * (G_max - G[y][x])) + w[y][x]
                            w[y][x] = 0 if tmp < 0 else tmp                            
                
                return (cv2.resize(G, (input.shape[1], input.shape[0]), fx=0, fy=0, interpolation=cv2.INTER_LINEAR)).astype(np.uint8)

            def incre_filling(h, original):
                h.astype(np.float32)
                original.astype(np.float32)

                height = h.shape[0]
                width = h.shape[1]
                w = np.zeros_like(h, np.float32)
                G = np.zeros_like(h, np.float32)

                for t in range(100):
                    G = w + h
                    for y in range(1, height-2):
                        for x in range(1, width-2):
                            tmp = 0.2 * (-G[y][x] + G[y+1][x]
                                        + -G[y][x] + G[y-1][x]
                                        + -G[y][x] + G[y][x+1]
                                        + -G[y][x] + G[y][x-1]) + w[y][x]
                            w[y][x] = 0 if tmp < 0 else tmp

                return (0.85 * original / G * 255).astype(np.uint8)

            imageYCrCb = cv2.split(cv2.cvtColor(self.orgImage, cv2.COLOR_RGB2YCrCb))
            Y = incre_filling(water_filling(imageYCrCb[0]), imageYCrCb[0])
            self.tmpImage = cv2.cvtColor(cv2.merge((Y, imageYCrCb[1], imageYCrCb[2])), cv2.COLOR_YCrCb2RGB)
            self.updateQueue()
            self.updateImageLabel(self.image)

    def mosaicButtonClicked(self):
        if self.pixmap:
            self.button = 3
            self.imageLabel.setCursor(QtGui.QCursor(QtCore.Qt.OpenHandCursor))

    def runMosaicButton(self):
        w, h = abs(self.x1 - self.x2), abs(self.y1 - self.y2)
        if w >= 15 and h >= 15:
            src = self.image.copy()
            roi = src[min(self.y1, self.y2):max(self.y1, self.y2), min(self.x1, self.x2):max(self.x1, self.x2)]
            roi = cv2.resize(roi, (w//15, h//15))
            roi = cv2.resize(roi, (w, h), interpolation=cv2.INTER_AREA)
            src[min(self.y1, self.y2):max(self.y1, self.y2), min(self.x1, self.x2):max(self.x1, self.x2)] = roi
            cv2.copyTo(src, self.mask, self.tmpImage)
            self.updateQueue()
            self.updateImageLabel(self.image)

    def correctionButtonClicked(self):
        if self.pixmap:
            self.button = -1
            self.imageLabel.setCursor(QtGui.QCursor(QtCore.Qt.ArrowCursor))
            ## 이미지 평활화
            y, cr, cb = cv2.split(cv2.cvtColor(self.orgImage.copy(), cv2.COLOR_RGB2YCrCb))
            self.tmpImage = cv2.cvtColor(cv2.merge([cv2.equalizeHist(y), cr, cb]), cv2.COLOR_YCrCb2RGB)
            ## 이미지 선명하게
            kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
            self.tmpImage = cv2.filter2D(self.tmpImage, -1, kernel)
            self.updateQueue()
            self.updateImageLabel(self.image)

    def edgeButtonClicked(self):
        if self.pixmap:
            self.button = -1
            self.imageLabel.setCursor(QtGui.QCursor(QtCore.Qt.ArrowCursor))
            blurImage = cv2.GaussianBlur(self.orgImage, (5, 5), cv2.BORDER_DEFAULT)
            cannyImage = 255 - cv2.Canny(blurImage, 30, 60)
            self.tmpImage = cv2.cvtColor(cannyImage, cv2.COLOR_GRAY2RGB)
            self.updateQueue()
            self.updateImageLabel(self.image)

    def webtoonButtonClicked(self):
        if self.pixmap:
            self.button = -1
            self.imageLabel.setCursor(QtGui.QCursor(QtCore.Qt.ArrowCursor))
            h, w = self.orgImage.shape[:2]
            resizeImage = cv2.resize(self.orgImage, (w//2, h//2))
            filterImage = cv2.bilateralFilter(resizeImage, -1, 20, 7)
            cannyImage = cv2.Canny(resizeImage, 80, 120)
            bitwiseImage = cv2.bitwise_and(filterImage, cv2.cvtColor(255 - cannyImage, cv2.COLOR_GRAY2RGB))
            self.tmpImage = cv2.resize(bitwiseImage, (w, h), interpolation=cv2.INTER_NEAREST)
            self.updateQueue()
            self.updateImageLabel(self.image)

    def sketchButtonClicked(self):
        if self.pixmap:
            self.button = -1
            self.imageLabel.setCursor(QtGui.QCursor(QtCore.Qt.ArrowCursor))
            grayImage = cv2.cvtColor(self.orgImage, cv2.COLOR_RGB2GRAY)
            blurImage = cv2.GaussianBlur(grayImage, (0, 0), 5)
            divideImage = cv2.divide(grayImage, blurImage, scale=255)
            self.tmpImage = cv2.cvtColor(divideImage, cv2.COLOR_GRAY2RGB)
            self.updateQueue()
            self.updateImageLabel(self.image)

    def undoButtonClicked(self):
        if self.undoQueue:
            self.redoQueue.appendleft(self.image)
            if not self.redoButton.isEnabled():
                self.redoButton.setEnabled(True)
            self.image = self.undoQueue.pop()
            if not self.undoQueue:
                self.undoButton.setEnabled(False)
            self.updateImageLabel(self.image)

    def redoButtonClicked(self):
        if self.redoQueue:
            self.undoQueue.append(self.image)
            if not self.undoButton.isEnabled():
                self.undoButton.setEnabled(True)
            self.image = self.redoQueue.popleft()
            if not self.redoQueue:
                self.redoButton.setEnabled(False)
            self.updateImageLabel(self.image)

    def returnButtonClicked(self):
        self.image = self.orgImage.copy()
        self.updateImageLabel(self.image)
        self.undoQueue.clear()
        self.redoQueue.clear()
        self.initUi()

    def rValueChanged(self):
        if self.pixmap:
            r, g, b = cv2.split(self.image)
            self.tmpImage = cv2.merge((r+(cv2.split(self.orgImage)[0][0]-r[0]+self.rTrackbar.value()), g, b))
            self.updateImageLabel(self.tmpImage)

    def gValueChanged(self):
        if self.pixmap:
            r, g, b = cv2.split(self.image)
            self.tmpImage = cv2.merge((r, g+(cv2.split(self.orgImage)[1][0]-g[0]+self.gTrackbar.value()), b))
            self.updateImageLabel(self.tmpImage)

    def bValueChanged(self):
        if self.pixmap:
            r, g, b = cv2.split(self.image)
            self.tmpImage = cv2.merge((r, g, b+(cv2.split(self.orgImage)[2][0]-b[0]+self.bTrackbar.value())))
            self.updateImageLabel(self.tmpImage)

    def trackbarReleased(self):
        if self.pixmap:
            self.updateQueue()

    def updateImageLabel(self, image):
        self.pixmap = QPixmap.fromImage(QImage(image.data, image.shape[1], image.shape[0], QImage.Format_RGB888))
        self.imageLabel.setPixmap(self.pixmap)

    def updateQueue(self):
        self.undoQueue.append(self.image)
        self.image = self.tmpImage.copy()
        self.redoQueue.clear()
        self.redoButton.setEnabled(False)
        if not self.undoButton.isEnabled():
            self.undoButton.setEnabled(True)
        if not self.returnButton.isEnabled():
            self.returnButton.setEnabled(True)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    mainWindow = MainWindow()
    mainWindow.show()
    sys.exit(app.exec_())