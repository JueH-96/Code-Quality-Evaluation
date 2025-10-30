# YOUR CODE HERE
n = int(input())
a = list(map(int, input().split()))

# Two pointer approach
# i points to potential smaller mochi
# j points to potential larger mochi
i = 0
j = 0
count = 0

# We need to find pairs where a[i] <= a[j]/2, which means a[i] * 2 <= a[j]
while i < n and j < n:
    # Skip if we're looking at the same position
    if i >= j:
        j = i + 1
    
    # If we've exhausted all possible larger mochi
    if j >= n:
        break
    
    # Check if we can make a kagamimochi with a[i] on top of a[j]
    if a[i] * 2 <= a[j]:
        # We found a valid pair
        count += 1
        i += 1
        j += 1
    else:
        # Current j is too small, try next larger mochi
        j += 1

print(count)