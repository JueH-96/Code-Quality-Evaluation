N = int(input())
S = list(input())
Q = int(input())

for _ in range(Q):
    t, x, c = input().split()
    t = int(t)
    x = int(x)
    
    if t == 1:
        S[x-1] = c
    elif t == 2:
        S = [ch.lower() for ch in S]
    else:
        S = [ch.upper() for ch in S]

print(''.join(S))