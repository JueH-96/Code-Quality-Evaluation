SAB, SAC, SBC = input().split()
if SAB == '<':
    if SAC == '<':
        print('B')
    elif SBC == '>':
        print('B')
    else:
        if SBC == '<':
            print('C')
        else:
            print('A')
else:
    if SAC == '>':
        print('B')
    elif SBC == '<':
        print('B')
    else:
        if SBC == '>':
            print('C')
        else:
            print('A')