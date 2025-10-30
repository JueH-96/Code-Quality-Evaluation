n = int(input())
d = list(map(int, input().split()))

ans = 0
for i in range(1, n + 1):
    if 10 > i:
        if 10 > d[i - 1]:
            ans += 1
        for j in range(10, min(d[i - 1] + 1, 100), 11):
            ans += 1
    else:
        if 100 > d[i - 1] and 100 > i and str(i)[0] == str(i)[1]:
            ans += 1
print(ans)