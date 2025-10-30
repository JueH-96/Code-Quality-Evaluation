# YOUR CODE HERE
n, q = map(int, input().split())
s = input().strip()

# Create array where consecutive[i] = 1 if s[i] == s[i+1]
consecutive = [0] * n
for i in range(n - 1):
    if s[i] == s[i + 1]:
        consecutive[i] = 1

# Build prefix sum array
prefix_sum = [0] * (n + 1)
for i in range(n):
    prefix_sum[i + 1] = prefix_sum[i] + consecutive[i]

# Answer queries
for _ in range(q):
    l, r = map(int, input().split())
    # Convert to 0-indexed
    l -= 1
    r -= 1
    
    # Count positions p where l <= p <= r-1 and s[p] == s[p+1]
    # This is the sum of consecutive[l] to consecutive[r-1]
    result = prefix_sum[r] - prefix_sum[l]
    print(result)