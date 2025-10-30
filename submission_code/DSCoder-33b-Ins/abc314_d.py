# YOUR CODE HERE

N = int(input())
S = list(input())
Q = int(input())

for _ in range(Q):
    t, x, c = input().split()
    t, x = int(t), int(x)
    if t == 1:
        S[x-1] = c
    elif t == 2:
        S = list(''.join(S).lower())
    elif t == 3:
        S = list(''.join(S).upper())

print(''.join(S))