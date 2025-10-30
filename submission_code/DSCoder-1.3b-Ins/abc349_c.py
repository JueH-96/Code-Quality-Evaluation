# YOUR CODE HERE

S = input()
T = input()

if T == S[:3].upper() and (len(T) == 3 or T == S[:2].upper() + 'X'):
    print('Yes')
else:
    print('No')