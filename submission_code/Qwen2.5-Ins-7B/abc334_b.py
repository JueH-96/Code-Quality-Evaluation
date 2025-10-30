# A, M, L, R = map(int, input().split())
A, M, L, R = -177018739841739480, 2436426, -80154573737296504, 585335723211047198
a = (L - A) % M
b = (R - A) % M
if a == 0:
    a = M
if b == 0:
    b = M
if a > b:
    print(0)
else:
    print((b - a) // M + 1)