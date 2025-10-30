from itertools import groupby

N = int(input())
S = input()
T = input()

def f(S):
    res = []
    for k, g in groupby(S):
        res.append((k, len(list(g))))
    return res

s = f(S)
t = f(T)

if len(s) != len(t):
    print(-1)
    exit()

for i in range(len(s)):
    if s[i][0] != t[i][0]:
        print(-1)
        exit()

    if s[i][1] > t[i][1]:
        print(-1)
        exit()

ans = 0
for i in range(len(s)):
    if s[i][1] < t[i][1]:
        ans += (t[i][1] - s[i][1]) * 2
        if i > 0:
            ans += 2

print(ans)