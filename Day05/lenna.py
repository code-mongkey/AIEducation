import turtle
import struct
inFp = open("images/lenna_64_Grey.bmp", "rb")

aa = [1,2,3,4,5,6,7,8,9]
print(aa[3:4])


bmpHeader = inFp.read(14)
bmpInfoHeader = inFp.read(40)

biOffBits=struct.unpack("i", bmpHeader[10:14])[0]
biWidth=struct.unpack("i", bmpInfoHeader[4:8])[0]
biHeight=struct.unpack("i", bmpInfoHeader[8:12])[0]
biBitCount=struct.unpack("H", bmpInfoHeader[14:16])[0]
biSizeImage=struct.unpack("i", bmpInfoHeader[20:24])[0]

inFp.seek(biOffBits)
bmpData = inFp.read(biSizeImage)

print("이진 데이터 시작 위치 : " + str(biOffBits))
print("가로 해상도 : " + str(biWidth))
print("세로 해상도 : " + str(biHeight))
print("픽셀당 bit 크기 : " + str(biBitCount))
print("이미지 데이터 크기 : " + str(biSizeImage))
print(bmpData)

turtle.speed(0)
for y in range(0, biHeight):
    turtle.goto(0,y)
    turtle.pendown()
    for x in range(0, biWidth):
        dotColor = struct.unpack('<B',bmpData[y*biWidth+x: y*biWidth+x+1])[0]
        turtle.goto(x,y)
        turtle.pencolor("#%02x%02x%02x" % (dotColor, dotColor, dotColor))
    turtle.penup()
    pass

turtle.done()
