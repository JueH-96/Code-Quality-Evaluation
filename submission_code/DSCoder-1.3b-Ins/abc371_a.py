# YOUR CODE HERE

S_AB, S_AC, S_BC = map(str, input().split())

if S_AB == '<':
    if S_AC == '<':
        print('A')
    else:
        print('B')
else:
    if S_AC == '>':
        print('C')
    else:
        print('A')