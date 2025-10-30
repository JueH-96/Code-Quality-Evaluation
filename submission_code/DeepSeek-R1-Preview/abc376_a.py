n, c = map(int, input().split())
times = list(map(int, input().split()))
count = 1
last = times[0]
for i in range(1, n):
    if times[i] - last >= c:
        count += 1
        last = times[i]
print(count)