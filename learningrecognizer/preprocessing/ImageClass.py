class Image:

    def __init__(self, classLabel, fname, img):

        self.classLabel = classLabel
        self.fname = fname
        self.img = img

    def setclassLabel(self, classLabel):
        self.classLabel = classLabel

    def setfname(self, fname):
        self.fname = fname

    def setimage(self, img):
        self.img = img

    def getclassLabel(self):
        return self.classLabel

    def getfname(self):
        return self.fname

    def getimage(self):
        return self.image