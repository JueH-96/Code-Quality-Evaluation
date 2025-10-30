# YOUR CODE HERE
S = input()

if all(S[i] == '0' for i in range(2, 16, 2)):
    print('Yes')
else:
    print('No')