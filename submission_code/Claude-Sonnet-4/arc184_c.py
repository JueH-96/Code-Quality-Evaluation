def is_mountain(pos):
    # pos is 1-indexed position of crease
    # Convert to binary and analyze
    pos_binary = pos
    
    # Find the rightmost 1 bit position
    rightmost_one_pos = 0
    temp = pos_binary
    while temp > 0:
        if temp & 1:
            break
        rightmost_one_pos += 1
        temp >>= 1
    
    # Count 1s to the left of the rightmost 1
    remaining = pos_binary >> (rightmost_one_pos + 1)
    count_ones = bin(remaining).count('1') if remaining > 0 else 0
    
    # Mountain fold if count is even
    return count_ones % 2 == 0

# Read input
n = int(input())
a = list(map(int, input().split()))

max_count = 0

# Since the numbers can be very large, we need to be smart about iteration
# We'll check a reasonable range and look for patterns
limit = min(10**6, 2**100 - a[-1])

for i in range(1, limit):
    count = 0
    for k in range(n):
        pos = i + a[k]
        if is_mountain(pos):
            count += 1
    max_count = max(max_count, count)

print(max_count)