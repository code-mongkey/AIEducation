import turtle
import random

from tkinter.simpledialog import *

inStr = ""
swidth, sheight = 500, 500
tX, tY, txtSize = [0] * 3

turtle.title('거북이 글자쓰기')
turtle.shape('turtle')
turtle.setup(swidth * 1.1, sheight * 1.1)
turtle.screensize(swidth,sheight)
turtle.penup()
turtle.speed(0.1)

inStr=askstring('문자열', '문자열을 입력하세요')
for ch in inStr:
    tX=random.randrange(-swidth / 2, swidth / 2)
    tY=random.randrange(-sheight / 2, sheight / 2)
    r=random.randrange(0,255)
    g=random.randrange(0,255)
    b=random.randrange(0,255)

    if turtle.penup():
        turtle.pendown()
    else:
        turtle.penup()

    turtle.goto(tX,tY)

    turtle.colormode(255)
    turtle.pencolor((r,g,b))
    turtle.write(ch, font=("맑은고딕", txtSize, "bold"))

turtle.done