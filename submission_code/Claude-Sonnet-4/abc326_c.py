import bisect

n, m = map(int, input().split())
a = list(map(int, input().split()))

# Sort the gift positions
a.sort()

max_gifts = 0

# Try each gift position as the left boundary of the interval
for i in range(n):
    left = a[i]
    right = left + m
    
    # Find the first position >= right using binary search
    right_idx = bisect.bisect_left(a, right)
    
    # Count gifts in [left, right) starting from position i
    count = right_idx - i
    
    max_gifts = max(max_gifts, count)

print(max_gifts)