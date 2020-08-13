def func1():
    global a
    a = 10
    print("func1의 a : "+str(a))

def func2():
    print("func2의 a : "+str(a))

a=20

func1()
func2()