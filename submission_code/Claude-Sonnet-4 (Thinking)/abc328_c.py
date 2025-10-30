n, q = map(int, input().split())
s = input().strip()

# Create array to mark consecutive identical characters
consecutive = [0] * n
for i in range(n - 1):
    if s[i] == s[i + 1]:
        consecutive[i] = 1

# Create prefix sum array
prefix = [0] * (n + 1)
for i in range(n):
    prefix[i + 1] = prefix[i] + consecutive[i]

# Answer queries
for _ in range(q):
    l, r = map(int, input().split())
    if r > l:
        # We want positions p where l <= p <= r-1 (1-indexed)
        # Convert to 0-indexed: (l-1) <= p <= (r-2)
        result = prefix[r - 1] - prefix[l - 1]
    else:
        result = 0
    print(result)