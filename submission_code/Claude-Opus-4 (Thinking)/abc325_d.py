n = int(input())
intervals = []
for i in range(n):
    t, d = map(int, input().split())
    intervals.append((t, t + d))  # (start_time, end_time)

# Sort by end time
intervals.sort(key=lambda x: x[1])

last_print_time = -1
count = 0

for start, end in intervals:
    # Find the earliest time we can print on this product
    # It must be at least start time and at least last_print_time + 1
    earliest_print = max(start, last_print_time + 1)
    
    if earliest_print <= end:
        # We can print on this product
        count += 1
        last_print_time = earliest_print

print(count)