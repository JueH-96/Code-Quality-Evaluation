import bisect

# Read input
N, M, P = map(int, input().split())
A = list(map(int, input().split()))
B = list(map(int, input().split()))

# Sort B for binary search
B.sort()

# Calculate prefix sum of B for efficient sum calculation
B_prefix = [0] * (M + 1)
for i in range(M):
    B_prefix[i + 1] = B_prefix[i] + B[i]

total = 0

for a in A:
    # Find how many side dishes b satisfy a + b < P
    # i.e., b < P - a
    threshold = P - a
    
    # Binary search to find the count
    count = bisect.bisect_left(B, threshold)
    
    # For the first 'count' side dishes, the price is a + b
    # Sum of these prices = count * a + sum of first 'count' B values
    if count > 0:
        total += count * a + B_prefix[count]
    
    # For the remaining (M - count) side dishes, the price is P
    remaining = M - count
    total += remaining * P

print(total)