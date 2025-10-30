n = int(input())
a = list(map(int, input().split()))

# Precompute left[i] = number of distinct elements in a[0:i+1]
left = [0] * n
seen = set()
for i in range(n):
    seen.add(a[i])
    left[i] = len(seen)

# Precompute right[i] = number of distinct elements in a[i:n]
right = [0] * n
seen = set()
for i in range(n-1, -1, -1):
    seen.add(a[i])
    right[i] = len(seen)

max_sum = 0

# Try all valid splits (i, j) where 1 <= i < j <= n-1
for i in range(1, n-1):  # i from 1 to n-2
    for j in range(i+1, n):  # j from i+1 to n-1
        # Left subarray: a[0:i] -> left[i-1]
        left_count = left[i-1]
        
        # Middle subarray: a[i:j] 
        middle_seen = set()
        for k in range(i, j):
            middle_seen.add(a[k])
        middle_count = len(middle_seen)
        
        # Right subarray: a[j:n] -> right[j]
        right_count = right[j]
        
        total = left_count + middle_count + right_count
        max_sum = max(max_sum, total)

print(max_sum)