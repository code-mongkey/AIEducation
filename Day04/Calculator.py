def calc(v1, v2, op):
    if op == '+':
        return v1 + v2
        pass
    elif op == '-':
        return v1 - v2
        pass
    elif op == '/':
        if v2 == 0: return "분모에 0이 올 수 없습니다"
        return v1 / v2
        pass
    elif op == '*':
        return v1 * v2
        pass
    else:
        return "계산식을 다시한번 확인해주세요"
        pass

