with open('result.txt','r') as f:
    lines = f.readlines()
    with open('result_2.json','w') as f:
        f.write('{\n')
        for line in lines:
            f.write(line.replace('\n', ',\n'))
        f.write('}')