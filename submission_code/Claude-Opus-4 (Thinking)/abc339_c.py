n = int(input())
a = list(map(int, input().split()))

# Calculate prefix sums and find the minimum
min_prefix = 0
current_sum = 0
for i in range(n):
    current_sum += a[i]
    min_prefix = min(min_prefix, current_sum)

# Minimum initial passengers needed
x_min = max(0, -min_prefix)

# Current passengers after all stops
current_passengers = x_min + current_sum

print(current_passengers)