def is_mountain(pos):
    if pos <= 0:
        return False
    # Count trailing zeros in binary representation
    trailing_zeros = (pos & -pos).bit_length() - 1
    # Check the bit after the trailing zeros
    return ((pos >> (trailing_zeros + 1)) & 1) == 1

n = int(input())
a = list(map(int, input().split()))

max_count = 0

# Check a range of starting positions
# Since the pattern has periodicities, we don't need to check the full range
upper_limit = min(2**20, 2**100 - a[-1]) if a[-1] < 2**100 else 0

for i in range(1, upper_limit + 1):
    count = sum(1 for offset in a if is_mountain(i + offset))
    max_count = max(max_count, count)

print(max_count)