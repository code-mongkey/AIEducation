dic = {1:"a",2:"c",3:"d",4:"a",5:"bb"}

for v in dic:
    print(v)


names = ['유비', '관우', '장비', '제갈공명', '박혁거세', '이순신', '홍길동']

count = 0
for name in names:
    if len(name) == 2:
        count += 1
        pass
    elif len(name) == 3:
        print("3글자인사람 : " + name)
        pass
    elif len(name) == 4:
        print("4글자인사람 : " + name)
        pass
    else:
        pass

print("2글자인 사람의 수 : " + str(count))


