import turtle as t
import random as r

def drawTree(length, angle):
	if length < 10:
		return
	
	t.pensize(length * 0.1)
	t.forward(length)
	
	t.right(angle)
	drawTree(length * r.uniform(0.5, 0.9), r.randint(10, 50))
	t.left(angle * 2)
	drawTree(length * r.uniform(0.5, 0.9), r.randint(10, 50))
	t.right(angle)
	
	t.pensize(length * 0.1)
	t.backward(length)

t.speed(0)
t.penup()
t.setpos(0, -200)
t.pendown()
t.left(90)

drawTree(100, r.randint(10, 40))

t.done()
