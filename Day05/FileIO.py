def readfile(path):
    inFp, inStr, ret = None, "", ""
    inFp = open(path, "r")
    inStr=inFp.readlines()
    for i in inStr:
        ret += i
    inFp.close()
    return ret

def writefile(path, outStr):
    outFp = None
    outFp = open(path, "w")
    outFp.writelines(outStr + "\n")
    outFp.close()
    pass

path = "C:/Temp/hello.txt"



while True:
    outStr = input("내용입력 : ")
    if outStr != "":
        writefile(path, outStr)
    else:
        break

print(readfile(path))


