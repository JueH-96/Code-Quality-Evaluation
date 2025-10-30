# coding: utf-8
# Your code here!
D = int(input())
L = int(D**.5)
ans = 10**18
for i in range(L+1):
    f = pow(i, 2)
    gloss = D - f
    y = int(gloss**.5)
    for j in range(y-1, y+2):
        if j*j+ f >= D:
            ans = min(ans, abs(D - f - j*j))

print(ans)