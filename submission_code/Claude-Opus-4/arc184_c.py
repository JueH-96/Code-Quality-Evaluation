def count_trailing_zeros(n):
    if n == 0:
        return 0
    count = 0
    while n & 1 == 0:
        n >>= 1
        count += 1
    return count

def is_mountain_fold(position):
    # Count trailing zeros in binary representation
    trailing_zeros = count_trailing_zeros(position)
    # Mountain fold if trailing zeros is odd
    return trailing_zeros % 2 == 1

# Read input
N = int(input())
A = list(map(int, input().split()))

# Maximum position we need to check
max_pos = 2**100 - A[-1] - 1

# Find maximum f(i)
max_mountain_count = 0

# We need to find the maximum number of mountain folds
# Since we're looking for patterns, we can optimize by checking specific ranges
# The pattern repeats with period related to powers of 2

# For each possible offset i
for i in range(1, min(max_pos + 1, 10**6)):  # Limit search for efficiency
    mountain_count = 0
    valid = True
    
    for a in A:
        pos = i + a
        if pos >= 2**100:
            valid = False
            break
        if is_mountain_fold(pos):
            mountain_count += 1
    
    if valid:
        max_mountain_count = max(max_mountain_count, mountain_count)

print(max_mountain_count)