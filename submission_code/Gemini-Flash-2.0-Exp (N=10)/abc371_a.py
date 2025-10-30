s_ab, s_ac, s_bc = input().split()

if s_ab == '<' and s_ac == '<':
    if s_bc == '<':
        print('B')
    else:
        print('C')
elif s_ab == '<' and s_ac == '>':
    print('A')
elif s_ab == '>' and s_ac == '<':
    print('A')
elif s_ab == '>' and s_ac == '>':
    if s_bc == '<':
        print('C')
    else:
        print('B')