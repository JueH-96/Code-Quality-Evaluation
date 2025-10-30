n = int(input())
products = []
for _ in range(n):
    t, d = map(int, input().split())
    end = t + d
    products.append((end, t))

products.sort(key=lambda x: (x[0], x[1]))

prev_time = -1
count = 0

for end, t in products:
    current_time = max(t, prev_time + 1)
    if current_time <= end:
        count += 1
        prev_time = current_time

print(count)