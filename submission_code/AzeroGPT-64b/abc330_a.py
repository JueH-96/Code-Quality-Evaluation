n, l = map(int, input().split())
a = list(map(int, input().split()))
cnt = 0
for i in a:
    if i >= l:
        cnt += 1
print(cnt)