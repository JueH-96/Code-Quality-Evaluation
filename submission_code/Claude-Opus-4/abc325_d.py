# YOUR CODE HERE
n = int(input())
intervals = []
for i in range(n):
    t, d = map(int, input().split())
    intervals.append((t, t + d))

# Sort intervals by end time
intervals.sort(key=lambda x: x[1])

# Greedy algorithm
last_print_time = -1
count = 0

for start, end in intervals:
    # Can we print this product?
    # We need to print at time >= max(start, last_print_time + 1)
    earliest_print = max(start, last_print_time + 1)
    
    if earliest_print <= end:
        # We can print this product
        count += 1
        # Print as late as possible to maximize future options
        last_print_time = end

print(count)