N = int(input())
WX = [list(map(int, input().split())) for _ in range(N)]

ans = 0
for i in range(24):
    cnt = 0
    for w, x in WX:
        if 9 <= (x - i) % 24 <= 18:
            cnt += w
    ans = max(ans, cnt)
print(ans)