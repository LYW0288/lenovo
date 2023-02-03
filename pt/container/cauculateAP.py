import os
import base64
import json
import requests
import cv2
import time
import pandas as pd
import xml.etree.ElementTree as ET
from yoloXml import yoloXmlClass
from io import StringIO

img_size = 320
def readLabelXMLToDF(inputDirect):
    readcount = 0
    errorXml = 0
    readImageOKcount = 0
    errorReadImg = 0
    totalobj = 0
    transformDictList = []
    
    groundTruthDF = pd.DataFrame(columns = ['filename','gbox','x1', 'y1' ,'x2', 'y2', 'check_result'])
    
    imageNameDict = {}
    for file in os.listdir(inputDirect):
        if file.endswith('.xml'):
            readcount +=1 
            print(f'readcount{readcount}')
            ##parse xml
            try:
                transformDict = {}
                tree = ET.parse(os.path.join(inputDirect, file))
                annotation = tree.getroot()
                
                transformDict['inputDirect'] = inputDirect
                transformDict['xmlname'] = file
                transformDict['imagename'] = file.replace('xml','jpg')
                transformDict['size'] = {}
                transformDict['size']['width'] = int(annotation.find('size').find('width').text)
                transformDict['size']['height'] = int(annotation.find('size').find('height').text)
                transformDict['size']['depth']  = int(annotation.find('size').find('depth').text)
                
                img = cv2.imread( os.path.join( transformDict['inputDirect'], transformDict['imagename'] ))
                if img is not None:
                    imageNameDict[ transformDict['imagename'] ] = transformDict
                    readImageOKcount += 1
                    
                    xmlObjList = annotation.findall('object')
                    # transformDict['objlist'] = []
                    for xmlObj in xmlObjList:
                        totalobj += 1
                        data = pd.DataFrame.from_dict({'filename':  transformDict['imagename'], 'gbox': xmlObj.find('name').text, 
                            'x1': int(xmlObj.find('bndbox').find('xmin').text), 'y1': int(xmlObj.find('bndbox').find('ymin').text),
                            'x2': int(xmlObj.find('bndbox').find('xmax').text), 'y2': int(xmlObj.find('bndbox').find('ymax').text),
                            'check_result': False}, orient = "index").T
                        groundTruthDF = pd.concat([groundTruthDF, data]).reset_index(drop=True)
                        
                else:
                    print(f"error read img: { os.path.join( transformDict['inputDirect'], transformDict['imagename'])}")
                    errorReadImg += 1
                    
            except Exception as e:
                print(e)
                errorXml += 1
    
    # print( imageNameDict[ transformDict['imagename'] ] )
    
    print('totalxml:', readcount)
    print('errorXml:',errorXml)
    print('ReadOkXML:', readImageOKcount)
    print('errorReadImg:',errorReadImg)
    print('totalobj:', totalobj)
    
    return groundTruthDF
    


def predictAndGetDF(inputDirect, ip, target_dir, cauculate, xml_dir):
    predictDF = pd.DataFrame(columns = ['filename','pbox','score','x1', 'y1' ,'x2', 'y2', 'check_result'])
    
    time_list = []
    for dirPath, dirNames, fileNames in os.walk( inputDirect):
        for innerFile in fileNames:
            if innerFile.endswith('.jpg'):
                print( os.path.join(dirPath, innerFile))
                if cauculate:
                    outXML = yoloXmlClass(innerFile, 'img_size' ,'img_size')
                    outXMLdir = innerFile.replace('.jpg','.xml') 
                
                with open( os.path.join( dirPath, innerFile), 'rb') as f:
                    byte_img = f.read()
                
                post = {}
                post['Item'] = innerFile
                post['Base64img'] = base64.b64encode(byte_img).decode('UTF-8')
                
                start_time = time.time()
                r = requests.post(url=f'http://{ip}:1234/predict', json=post, verify=False)
                response = json.loads(r.json())
                print(response)
                end_time = time.time()
                time_list.append((end_time - start_time))
                
                img_cv2 = cv2.imread( os.path.join( dirPath, innerFile))
                copycv2 = img_cv2.copy()
                text_size = int(max(img_cv2.shape)/img_size * 2)
                font_size = int(max(img_cv2.shape)/img_size / 2)
                
                # response={}
                # response['ObjectList'] = [
                    # {"Object": "A1", "Score": 99.18, "X1": 48, "Y1": 133, "X2": 188, "Y2": 186},
                    # {"Object": "A1", "Score": 88.32, "X1": 0, "Y1": 50, "X2": 188, "Y2": 186}
                # ]
                for obj in response['ObjectList']:
                    # cv2.rectangle( img_cv2, (obj['X1'], obj['Y1']), (obj['X2'], obj['Y2']), (0, 0, 255), text_size)
                    # cv2.putText(img_cv2, obj['Object'], (obj['X1'], obj['Y1']), cv2.FONT_HERSHEY_SIMPLEX, font_size, (255, 0, 0), font_size, cv2.LINE_AA)
                    if cauculate: outXML.addObject(obj['Object'], str(obj['X1']), str(obj['Y1']), str(obj['X2']), str(obj['Y2']))
                    data = pd.DataFrame.from_dict(
                        {'filename':  innerFile, 'pbox': obj['Object'], 'score': obj['Score'],
                        'x1': obj['X1'], 'y1': obj['Y1'], 'x2': obj['X2'], 'y2': obj['Y2'], 'check_result': False}, orient = "index").T
                    predictDF = pd.concat([predictDF, data]).reset_index(drop=True)
                if cauculate:
                    outXML.output( os.path.join(xml_dir, outXMLdir) )
                    cv2.imwrite( os.path.join(xml_dir, innerFile) , copycv2)
                # cv2.imwrite( os.path.join(target_dir, innerFile) , img_cv2)
                # time.sleep(0.5)
    return predictDF, time_list

def calculate_IoU(predicted_bound, ground_truth_bound):
    """
    computing the IoU of two boxes.
    Args:
        # box: (xmin, ymin, xmax, ymax),通过左下和右上两个顶点坐标来确定矩形位置
        box: (xmin, ymin, xmax, ymax),通过左上和右下两个顶点坐标来确定矩形位置
    Return:
        IoU: IoU of box1 and box2.
    """
    
    pxmin, pymin, pxmax, pymax = predicted_bound
    # print("预测框P的坐标是：({}, {}, {}, {})".format(pxmin, pymin, pxmax, pymax))
    gxmin, gymin, gxmax, gymax = ground_truth_bound
    # print("原标记框G的坐标是：({}, {}, {}, {})".format(gxmin, gymin, gxmax, gymax))
    
    parea = (pxmax - pxmin) * (pymax - pymin)  # 计算P的面积
    garea = (gxmax - gxmin) * (gymax - gymin)  # 计算G的面积
    # print("预测框P的面积是：{}；原标记框G的面积是：{}".format(parea, garea))

    # 求相交矩形的左下和右上顶点坐标(xmin, ymin, xmax, ymax)
    #xmin = max(pxmin, gxmin)  # 得到左下顶点的横坐标
    #ymin = max(pymin, gymin)  # 得到左下顶点的纵坐标
    #xmax = min(pxmax, gxmax)  # 得到右上顶点的横坐标
    #ymax = min(pymax, gymax)  # 得到右上顶点的纵坐标
    
    # 求相交矩形的左上和右下顶点坐标(xmin, ymin, xmax, ymax)
    xmin = max(pxmin, gxmin)  # 得到左上顶点的横坐标
    ymin = max(pymin, gymin)  # 得到左上顶点的纵坐标
    xmax = min(pxmax, gxmax)  # 得到右下顶点的横坐标
    ymax = min(pymax, gymax)  # 得到右下顶点的纵坐标
    
    # 计算相交矩形的面积
    w = xmax - xmin
    h = ymax - ymin
    if w <=0 or h <= 0:
        return 0
    
    area = w * h  # G∩P的面积
    # area = max(0, xmax - xmin) * max(0, ymax - ymin)  # 可以用一行代码算出来相交矩形的面积
    # print("G∩P的面积是：{}".format(area))

    # 并集的面积 = 两个矩形面积 - 交集面积
    IoU = area / (parea + garea - area)

    return IoU
   

def cauculateAP(image_dir, ip, json_dir, cauculate=False): 

    print('start...')

    if not os.path.isdir(f'{json_dir}/validation'):
        os.makedirs( f'{json_dir}/validation')
    if not os.path.isdir(f'{json_dir}/validation/testpic'):
        os.makedirs( f'{json_dir}/validation/testpic')
    if not os.path.isdir(f'{json_dir}/labelimg_xml_created_by_ai') and cauculate:
        os.makedirs( f'{json_dir}/labelimg_xml_created_by_ai')
    target_dir = f'{json_dir}/validation/testpic'
    predictDF, time_list = predictAndGetDF( image_dir, ip, target_dir, cauculate, f'{json_dir}/labelimg_xml_created_by_ai')
    #print(predictDF)

    groundTruthDF = readLabelXMLToDF( image_dir)
    #print(groundTruthDF)

    # check obj count from groundTruthDF and predictDF
    objCaculateDict = {}
    objnames = groundTruthDF['gbox'].unique()
    #print(objnames)

    for objname in objnames:
        objCaculateDict[objname] = {}
        objCaculateDict[objname]['ground_truth_count'] = len( groundTruthDF[ groundTruthDF['gbox'] == objname ])
        objCaculateDict[objname]['predict_count'] = len( predictDF[ predictDF['pbox'] == objname ])
        
    #print(objCaculateDict)

    # check predict obj hit
    for objname in objnames:
        # iterate all predict obj
        #print(f'begin to check {objname}, iou and score.................................................................')
        
        for pindex, prow in predictDF[ predictDF['pbox'] == objname ].iterrows():
            #print('\npredict box:')
            #print(pindex , prow['filename'], prow['pbox'], prow['score'], prow['x1'], prow['y1'], prow['x2'], prow['y2'])
            
            # check roi and name in GT
            for gindex, grow in  groundTruthDF[ (groundTruthDF['filename'] == prow['filename']) & (groundTruthDF['gbox'] == objname)].iterrows():
                #print('\nfind same file name and box...')
                #print(gindex , grow['filename'], grow['gbox'], grow['x1'], grow['y1'], grow['x2'], grow['y2'])
                
                IoU = calculate_IoU( 
                    ( prow['x1'], prow['y1'], prow['x2'], prow['y2']),
                    ( grow['x1'], grow['y1'], grow['x2'], grow['y2'])
                )
                # print(f'--------\nIOU: {IoU}\tpbound: {prow}\tgbound: {grow}\n========')
                #print(f'\ncauculate {IoU}...')
                if IoU >= 0.5:
                    #print('IoU > 0.5')
                    predictDF.loc[ pindex, ['check_result']] = True
                    groundTruthDF.loc[ gindex, ['check_result']] = True
                    
        objCaculateDict[objname]['predict_hit_count'] = len( predictDF[ (predictDF['pbox'] == objname) & (predictDF['check_result'] == True)] )
        objCaculateDict[objname]['ground_hit_count'] = len( groundTruthDF[ (groundTruthDF['gbox'] == objname) & (groundTruthDF['check_result'] == True)] )
    
    #print(objCaculateDict)

    for dirPath, dirNames, fileNames in os.walk( image_dir):
        for innerFile in fileNames:
            if innerFile.endswith('.jpg'):
                img_cv2 = cv2.imread( os.path.join( dirPath, innerFile))
                text_size = int(max(img_cv2.shape)/img_size * 2)
                font_size = int(max(img_cv2.shape)/img_size / 2)
                for pindex, prow in predictDF[predictDF['filename'] == innerFile].iterrows():
                    cv2.rectangle( img_cv2, (prow['x1'], prow['y1']), (prow['x2'], prow['y2']), (0, 255*(prow['check_result']), 255*(not prow['check_result'])), text_size)
                    cv2.putText(img_cv2, prow['pbox'], (prow['x1'], prow['y1']), cv2.FONT_HERSHEY_SIMPLEX, font_size, (255, 0, 0), font_size, cv2.LINE_AA)
                cv2.imwrite( os.path.join(target_dir, innerFile) , img_cv2)

    # cauculate obj recall and precision 
    for objname in objnames:
        if objCaculateDict[objname]['predict_count'] != 0:
            objCaculateDict[objname]['precision'] = objCaculateDict[objname]['predict_hit_count'] / objCaculateDict[objname]['predict_count']
        else:
            objCaculateDict[objname]['precision'] = 'Null'
        
        if objCaculateDict[objname]['ground_truth_count'] != 0:
            objCaculateDict[objname]['recall'] = objCaculateDict[objname]['ground_hit_count'] / objCaculateDict[objname]['ground_truth_count']
        else:
            objCaculateDict[objname]['recall'] = 'Null'
        
    #print(objCaculateDict)\
    for objname in objnames:
        print(f'[{objname}]')
        print(f"\tTRUTH_COUNT\t{objCaculateDict[objname]['ground_truth_count']}")
        print(f"\tPREDICT_COUNT\t{objCaculateDict[objname]['predict_count']}")
        print(f"\tpredict_hit_count\t{objCaculateDict[objname]['predict_hit_count']}")
        print(f"\tPRECISION\t{objCaculateDict[objname]['precision']}")
        print(f"\tRECALL\t\t{objCaculateDict[objname]['recall']}")\

    with open(f'{json_dir}/validation/result.json',  'w') as f:
        f.write( json.dumps( objCaculateDict, indent = 4))
    groundTruthDF.to_csv(f"{json_dir}/validation/ground.csv")
    predictDF.to_csv(f"{json_dir}/validation/predict.csv")
    print('_________________________')
    print('\t\tPRED NUM')
    print(':.......................:')
    for objname in objCaculateDict:
        print(f"[{objname}]: ", objCaculateDict[objname]['predict_count'], " items.")
    print('_________________________')
    print('\t\tAVG TIME')
    print(':.......................:')
    print(sum(time_list)/len(time_list))
    print("=========================")

