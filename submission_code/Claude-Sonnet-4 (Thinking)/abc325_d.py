n = int(input())
products = []
for i in range(n):
    t, d = map(int, input().split())
    products.append((t, t + d))

# Sort by start time
products.sort()

count = 0
last_print_time = -1

for start, end in products:
    earliest_print = max(start, last_print_time + 1)
    if earliest_print <= end:
        count += 1
        last_print_time = earliest_print

print(count)