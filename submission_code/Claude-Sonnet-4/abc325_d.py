n = int(input())
products = []
for i in range(n):
    t, d = map(int, input().split())
    products.append((t, t + d))  # (start_time, end_time)

# Sort by end time
products.sort(key=lambda x: x[1])

count = 0
last_print_time = -float('inf')

for start_time, end_time in products:
    # We want to print as late as possible, but at least 1 microsecond after last print
    earliest_print_time = max(start_time, last_print_time + 1)
    
    # Check if we can print within the product's time window
    if earliest_print_time <= end_time:
        # Print at the end time to maximize flexibility
        last_print_time = end_time
        count += 1

print(count)