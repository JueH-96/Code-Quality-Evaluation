X, Y, Z = map(int, input().split())
S = input()

ans = 0
prev = 0
for s in S:
    if prev == 0:
        if s == 'a':
            ans += min(X, Y)
        else:
            ans += min(X, Z) + Y
            prev = 1
    else:
        if s == 'a':
            ans += min(Y, Z) + X
            prev = 0
        else:
            ans += min(Y, X)
print(ans)