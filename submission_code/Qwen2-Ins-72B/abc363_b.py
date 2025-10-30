n, t, p = map(int, input().split())
l = list(map(int, input().split()))
l = [t-i for i in l]
l.sort()
ans = 0
cnt = 0
for i in l:
    if i <= 0:
        cnt += 1
    if cnt >= p:
        print(0)
        exit()
for i in l:
    ans += 1
    if i <= ans:
        cnt += 1
    if cnt >= p:
        print(ans)
        exit()