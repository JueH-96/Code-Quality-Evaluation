N = int(input())
A = list(map(int, input().split()))

# Find minimum value needed at start to keep non-negative at all points
min_start = 0
curr_sum = 0

# Go through each stop and track minimum value needed
for x in A:
    curr_sum += x
    min_start = max(min_start, -curr_sum)

print(min_start + curr_sum)