import Calculator

v1,v2,oper = 0,0,""

while(True):
    oper = input("계산을 입력하세요(+,-,*,/), 종료하시려면 c를 눌러주세요")
    if oper.upper() == "C": break
    
    v1 = int(input("첫번째 수를 입력하세요 : "))
    v2 = int(input("두번째 수를 입력하세요 : "))

    res = Calculator.calc(v1,v2,oper)

    print(res)
