# YOUR CODE HERE
n = int(input())
h = list(map(int, input().split()))

# For each position, we need to track when water first arrives
result = []

# The key insight is that water reaches position i when:
# - All previous positions have been filled to their barrier heights
# - Position i-1 has water level > H_i

# We can calculate this by considering the cumulative effect
# For position i to receive water, we need enough operations to:
# 1. Fill all positions 0 to i-1 to at least their barrier heights
# 2. Have one extra unit at position i-1 to overflow to position i

# Let's think about it differently:
# For water to reach position i for the first time, we need:
# - At least H_1 + 1 operations for water to reach position 1
# - At least max(H_1, H_2) + H_1 + 1 for position 2
# - And so on...

# Actually, let's simulate more carefully
# The water level at each position after k operations can be computed

# For each position i, we need to find the minimum number of operations
# such that A_i > 0

for target in range(1, n + 1):
    # Binary search for the minimum number of operations
    left, right = 1, 10**18
    
    while left < right:
        mid = (left + right) // 2
        
        # Simulate mid operations
        a = [0] * (n + 1)
        a[0] = mid
        
        # Perform the flow operations
        for i in range(1, n + 1):
            if a[i-1] > a[i] and a[i-1] > h[i-1]:
                transfer = min(a[i-1] - a[i], a[i-1] - h[i-1])
                a[i-1] -= transfer
                a[i] += transfer
        
        if a[target] > 0:
            right = mid
        else:
            left = mid + 1
    
    result.append(left)

print(' '.join(map(str, result)))