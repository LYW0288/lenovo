import xml.dom.minidom  

class yoloXmlClass:
    def __init__(self,input_filename, input_width ,input_height):
        self.__version__ = 'v0.0.1'

        self.doc = xml.dom.minidom.Document()
        self.annotation = self.doc.createElement('annotation')

        self.filename = self.doc.createElement('filename')
        self.filename_text = self.doc.createTextNode(input_filename)
        self.filename.appendChild(self.filename_text)

        self.size = self.doc.createElement('size')

        self.width = self.doc.createElement('width')
        self.width_text = self.doc.createTextNode(input_width)
        self.width.appendChild(self.width_text)

        self.height = self.doc.createElement('height')
        self.height_text = self.doc.createTextNode(input_height)
        self.height.appendChild(self.height_text)

        self.depth = self.doc.createElement('depth')
        self.depth_text = self.doc.createTextNode('3')
        self.depth.appendChild(self.depth_text)

        self.doc.appendChild(self.annotation)
        self.annotation.appendChild(self.filename)
        self.annotation.appendChild(self.size)

        self.size.appendChild(self.width)
        self.size.appendChild(self.height)
        self.size.appendChild(self.depth)

    def addObject(self, input_objmame, input_xmin, input_ymin, input_xmax, input_ymax):
        self.objectEle = self.doc.createElement('object')

        self.name = self.doc.createElement('name')
        self.name_text = self.doc.createTextNode(input_objmame)
        self.name.appendChild(self.name_text)

        self.difficult = self.doc.createElement('difficult')
        self.difficult_text = self.doc.createTextNode('0')
        self.difficult.appendChild(self.difficult_text)

        self.bndbox = self.doc.createElement('bndbox')

        self.xmin = self.doc.createElement('xmin')
        self.xmin_text = self.doc.createTextNode(input_xmin)
        self.xmin.appendChild(self.xmin_text)

        self.ymin = self.doc.createElement('ymin')
        self.ymin_text = self.doc.createTextNode(input_ymin)
        self.ymin.appendChild(self.ymin_text)

        self.xmax = self.doc.createElement('xmax')
        self.xmax_text = self.doc.createTextNode(input_xmax)
        self.xmax.appendChild(self.xmax_text)

        self.ymax = self.doc.createElement('ymax')
        self.ymax_text = self.doc.createTextNode(input_ymax)
        self.ymax.appendChild(self.ymax_text)

        #######################################################
        self.annotation.appendChild(self.objectEle)

        self.objectEle.appendChild(self.name)
        self.objectEle.appendChild(self.difficult)
        self.objectEle.appendChild(self.bndbox)

        self.bndbox.appendChild(self.xmin)
        self.bndbox.appendChild(self.ymin)
        self.bndbox.appendChild(self.xmax)
        self.bndbox.appendChild(self.ymax)

    def output(self, output_name):
        with open(output_name, 'w') as fw:
            self.doc.writexml(fw, indent='\t', addindent='\t', newl='\n', encoding="utf-8")
