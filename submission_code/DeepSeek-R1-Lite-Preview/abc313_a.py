N = int(input())
P = list(map(int, input().split()))

if N == 1:
    x = 0
else:
    P1 = P[0]
    others = P[1:]
    max_others = max(others)
    if P1 > max_others:
        x = 0
    else:
        x = max_others - P1 + 1
print(x)