# csv文件格式转化为libsvm文件格式
import pandas as pd
import time


def libsvm(df, fp):
    now = time.time()
    print('Format Converting begin in time：..........', now)
    columns_head = df.columns.values
    col_num = len(columns_head)
    feature_index = [i for i in range(col_num)]
    field = []
    for col in columns_head:
        field.append(str(col))

    with open(fp, 'w') as f:
        for row in df.values:
            line = str(row[0])
            for i in range(1, len(row)):
                line += " %s:%d" % (row[i], 1)
            line += '\n'
            f.write(line)
    print('finish convert,the cost time is ', time.time() - now)
    print('[Done]')


if __name__ == '__main__':

    df = pd.read_csv(r'/tmp/pycharm_project_dbreg/resources/dataset/frappe_csv/valid.csv')
    fp = r'/tmp/pycharm_project_dbreg/resources/dataset/frappe_csv/valid.libsvm'
    libsvm(df, fp)
