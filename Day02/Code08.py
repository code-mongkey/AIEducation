i,k = 0,0
size = 9

i = 0
while i < size:
    if i < size / 2:
        k = 0
        while k < size/2 - i:
            print(' ', end = '')
            k += 1
        k = 0
        while k < i * 2 + 1:
            print('\u2605', end='')
            k +=1
    else:
        k = 0
        while k < (i+1) - (size/2):
            print(' ', end = '')
            k += 1
        k = 0
        while k < (size - i) * 2 - 1:
            print('\u2605', end='')
            k +=1
            
    print()
    i += 1