n = int(input())
d = list(map(int, input().split()))
count = 0

for i in range(1, n + 1):
    di = str(i)
    if len(set(di)) != 1:
        continue
    unique_i = di[0]
    max_day = d[i - 1]
    for j in range(1, max_day + 1):
        dj = str(j)
        if len(set(dj)) != 1:
            continue
        if dj[0] == unique_i:
            count += 1

print(count)