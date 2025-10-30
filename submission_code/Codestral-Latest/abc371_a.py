# YOUR CODE HERE
S_AB, S_AC, S_BC = input().split()

if S_AB == '<' and S_AC == '<' and S_BC == '<':
    print('B')
elif S_AB == '<' and S_AC == '<' and S_BC == '>':
    print('C')
elif S_AB == '<' and S_AC == '>' and S_BC == '<':
    print('A')
elif S_AB == '<' and S_AC == '>' and S_BC == '>':
    print('B')
elif S_AB == '>' and S_AC == '<' and S_BC == '<':
    print('A')
elif S_AB == '>' and S_AC == '<' and S_BC == '>':
    print('C')
elif S_AB == '>' and S_AC == '>' and S_BC == '<':
    print('C')
elif S_AB == '>' and S_AC == '>' and S_BC == '>':
    print('B')