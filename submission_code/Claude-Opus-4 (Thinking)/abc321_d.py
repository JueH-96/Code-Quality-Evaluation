import bisect

n, m, p = map(int, input().split())
a = list(map(int, input().split()))
b = list(map(int, input().split()))

# Sort B array
b.sort()

# Compute prefix sum
prefix_sum = [0]
for x in b:
    prefix_sum.append(prefix_sum[-1] + x)

total = 0

for ai in a:
    # Find the number of B_j's such that ai + B_j <= p
    # This means B_j <= p - ai
    
    # Use bisect_right to find the position
    count = bisect.bisect_right(b, p - ai)
    
    # For the first 'count' elements, we add ai + b[j]
    total += count * ai + prefix_sum[count]
    
    # For the remaining (m - count) elements, we add p
    total += (m - count) * p

print(total)