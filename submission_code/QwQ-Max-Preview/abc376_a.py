n, c = map(int, input().split())
times = list(map(int, input().split()))
count = 0
last_time = -c  # Initialize to ensure the first press is counted
for t in times:
    if t - last_time >= c:
        count += 1
        last_time = t
print(count)