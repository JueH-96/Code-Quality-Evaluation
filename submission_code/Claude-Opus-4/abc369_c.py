# YOUR CODE HERE
n = int(input())
a = list(map(int, input().split()))

count = 0

# For each starting position
for l in range(n):
    # Single element is always an arithmetic progression
    count += 1
    
    if l == n - 1:
        continue
    
    # Find the longest arithmetic progression starting at position l
    # First, establish the common difference with the next element
    d = a[l + 1] - a[l]
    r = l + 1
    
    # Extend as long as the difference remains the same
    while r < n - 1 and a[r + 1] - a[r] == d:
        r += 1
    
    # Count all valid subarrays from l to any position between l+1 and r
    count += (r - l)

print(count)