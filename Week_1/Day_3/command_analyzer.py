import csv


def getKey(item):  # 정렬을위한함수
    return item[1]  # 신경쓸필요없음


command_data = []  # 파일읽어오기
with open("command_data.csv", "r", encoding='utf8') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',', quotechar='"')
    for row in spamreader:
        command_data.append(row)
command_counter = {}  # dict생성, 아이디를key값, 입력줄수를value값
for data in command_data:  # list 데이터를dict로변경
    if data[1] in command_counter.keys():  # 아이디가이미 Key값으로변경되었을때
        command_counter[data[1]] += 1  # 기존출현한아이디
    else:
        command_counter[data[1]] = 1  # 처음나온아이디
dictlist = []  # dict를list로변경
for key, value in command_counter.items():
    temp = [key, value]
    dictlist.append(temp)
sorted_dict = sorted(dictlist, key=getKey, reverse=True)  # list를입력줄수로정렬
print(sorted_dict[:100])  # List의상위10객값만출력
