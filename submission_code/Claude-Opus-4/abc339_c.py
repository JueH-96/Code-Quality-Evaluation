# YOUR CODE HERE
n = int(input())
a = list(map(int, input().split()))

# Calculate prefix sums and find the minimum
prefix_sum = 0
min_prefix_sum = 0

for i in range(n):
    prefix_sum += a[i]
    min_prefix_sum = min(min_prefix_sum, prefix_sum)

# Minimum initial passengers needed
initial_passengers = max(0, -min_prefix_sum)

# Current passengers = initial + total change
current_passengers = initial_passengers + sum(a)

print(current_passengers)