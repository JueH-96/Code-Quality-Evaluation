n, c = map(int, input().split())
times = list(map(int, input().split()))

count = 1
last_time = times[0]

for i in range(1, n):
    current_time = times[i]
    if current_time - last_time >= c:
        count += 1
        last_time = current_time

print(count)