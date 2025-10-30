s_ab, s_ac, s_bc = input().split()

if s_ab == '<':
    if s_ac == '<':
        if s_bc == '<':
            print('B')
        else:
            print('C')
    else:
        print('A')
else:
    if s_ac == '<':
        print('A')
    else:
        if s_bc == '<':
            print('C')
        else:
            print('B')