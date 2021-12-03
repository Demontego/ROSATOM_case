from eolearn.core import EOPatch
import numpy as np
import skimage.measure
import os
import joblib
import tqdm
import json
import sys
from lightgbm import LGBMClassifier

THRESH = 2.1488918011065634e-05

class FolderProcessor:
    
    def __init__(self, predictor, thresh):
        self.predictor = predictor
        self.thresh = thresh

    def process_dir(self, directory):
        jsons = []
        diffpack = np.unique([x.split('2')[0][:-1] for x in os.listdir(directory)])
        for pack in diffpack:
            eos = []
            full_square = 0
            for file in tqdm.tqdm([x for x in os.listdir(directory) if pack in x]):
                row = {}
                try:
                    eopatch = EOPatch.load(os.path.join(directory,file))
                    try:
                        lat, lon = eopatch.bbox.max_x,  eopatch.bbox.max_y
                    except:
                        continue
                    for img, t in zip(eopatch.data['L2A'], eopatch.timestamp):
                        img = img.reshape(13, 64, 64)
                        timestamp = t
                        vector = []
                        for channel in img:
                            mean = np.mean(channel)
                            std = np.std(channel)
                            feature = skimage.measure.block_reduce(
                                channel, (8, 8), np.max).reshape(8*8).tolist()
                            vector.append(mean)
                            vector.append(std)
                            vector.extend(feature)
                        if sum(vector) == 0:
                            continue
                        else:
                            pred = self.predictor.predict_proba(np.array(vector).reshape(1,-1))[0][1]
                            sqr = 0
                            if pred > self.thresh:
                                sqr = 64*64*10
                                full_square += sqr
                                row = {'lat': lat, 'lon': lon, 'timestamp': timestamp,'prob': pred}
                                eos.append(row)
                except:
                    continue
            if full_square==0:
                continue
            else:
                tss = sorted([x['timestamp'] for x in eos], reverse=True)
                eos = sorted(eos, key = lambda x: x['prob'])
                for i in range(len(eos)-1, -1, -1):
                    eos[i]['timestamp'] = tss[0].strftime('%d.%m.%Y')
                pack_data = {'name': pack, 'meanlat': np.mean([x['lat'] for x in eos]), 'meanlon': np.mean([x['lon'] for x in eos]),'timestamp':tss[0].strftime('%d.%m.%Y'), 'full_square': full_square,  'data': eos}
                jsons.append(pack_data)
        return jsons
    

if __name__ == '__main__':
    predictor = joblib.load(sys.argv[3])
    directory = sys.argv[1]
    output = sys.argv[2]
    fp = FolderProcessor(predictor, THRESH)
    datas = fp.process_dir(directory)
    with open(output, 'w', encoding='utf-8') as fout:
        json.dump(datas, fout)
    print('Successfully processed')
