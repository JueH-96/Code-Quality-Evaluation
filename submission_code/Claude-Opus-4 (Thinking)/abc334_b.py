# YOUR CODE HERE
A, M, L, R = map(int, input().split())

left = L - A
right = R - A

# Find the smallest integer k >= left / M (ceiling)
k_min = -(-left // M)

# Find the largest integer k <= right / M (floor)
k_max = right // M

# Count the number of integers from k_min to k_max
if k_min <= k_max:
    count = k_max - k_min + 1
else:
    count = 0

print(count)