# coding: utf-8
N = int(input())
S = input()
cnt = [0]*3
ans = 0
char = [0,1,2]
for i in range(N):
    c = S[i]
    if c == "A":
        c = 0
    elif c == "B":
        c = 1
    else:
        c = 2
    cnt[c] += 1
    char.remove(c)
    if len(char) == 0:
        ans = i+1
        break
print(ans)