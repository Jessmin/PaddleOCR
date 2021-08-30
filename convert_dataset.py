import json

output = '/home/zhaohj/Documents/dataset/Table/TAL/val_paddle.json'
out_f = open(output, 'w')
with open('/home/zhaohj/Documents/dataset/Table/TAL/val.json', 'r') as f:
    lines = f.readlines()
    for line in lines:
        data = json.loads(line)
        cells = data['html']['cells']
        for cell in cells:
            bbox = cell['bbox']
            _x = [x[0] for x in bbox]
            _y = [x[1] for x in bbox]
            bbox = [min(_x), min(_y), max(_x), max(_y)]
            cell['bbox'] = bbox
        out_f.write(json.dumps(data, ensure_ascii=False))
        out_f.write('\n')
