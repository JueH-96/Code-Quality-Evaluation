def is_mountain(n):
    """Returns True if n-th crease is a mountain fold."""
    while n % 2 == 0:
        n //= 2
    return n % 4 == 1

N = int(input())
A = list(map(int, input().split()))

# Check a sufficient range to capture the pattern
# Using 10^6 as it should be enough given the periodic nature
limit = min(10**6, (1 << 100) - A[-1] - 1)
result = 0

for i in range(1, limit + 1):
    count = sum(1 for a in A if is_mountain(i + a))
    result = max(result, count)

print(result)