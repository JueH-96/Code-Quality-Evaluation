from math import sqrt

n, d = map(int, input().split())
d = d**2
cor = [list(map(int, input().split())) for _ in range(n)]
imp = [0] * n
imp[0] = 1
stack = [0]
while stack:
    ci = stack.pop()
    c1 = cor[ci]
    for i, c2 in enumerate(cor):
        if not imp[i] and (c1[0]-c2[0])**2 + (c1[1] - c2[1])**2 <= d:
            imp[i] = 1
            stack.append(i)
print("
".join("Yes" if i else "No" for i in imp))