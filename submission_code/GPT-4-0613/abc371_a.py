# YOUR CODE HERE
s_ab, s_ac, s_bc = input().split()

if s_ab == '<':
    if s_ac == '<':
        print('B')
    else:
        print('C')
else:
    if s_ac == '>':
        print('A')
    else:
        print('B')