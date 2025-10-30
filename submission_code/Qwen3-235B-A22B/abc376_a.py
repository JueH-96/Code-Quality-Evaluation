n, c = map(int, input().split())
t = list(map(int, input().split()))

count = 1
last_time = t[0]

for i in range(1, n):
    current_time = t[i]
    if current_time - last_time >= c:
        count += 1
        last_time = current_time

print(count)