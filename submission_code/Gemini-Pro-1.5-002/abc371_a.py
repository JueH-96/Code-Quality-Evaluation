# YOUR CODE HERE
s_ab, s_ac, s_bc = input().split()

if s_ab == '<' and s_bc == '<':
    print('B')
elif s_ab == '>' and s_bc == '>':
    print('B')
elif s_ab == '<' and s_ac == '>':
    print('C')
elif s_ab == '>' and s_ac == '<':
    print('C')
elif s_ac == '<' and s_bc == '>':
    print('A')
elif s_ac == '>' and s_bc == '<':
    print('A')