# YOUR CODE HERE
n, q = map(int, input().split())
s = input()

# Precompute prefix sums for 1's and suffix sums for 2's
ones_prefix = [0] * (n + 1)
twos_suffix = [0] * (n + 1)

for i in range(n):
    ones_prefix[i + 1] = ones_prefix[i] + (1 if s[i] == '1' else 0)

for i in range(n - 1, -1, -1):
    twos_suffix[i] = twos_suffix[i + 1] + (1 if s[i] == '2' else 0)

for _ in range(q):
    l, r = map(int, input().split())
    l -= 1  # Convert to 0-indexed
    
    max_length = 0
    
    for i in range(l, r):
        if s[i] == '/':
            ones_before = ones_prefix[i] - ones_prefix[l]
            twos_after = twos_suffix[i + 1] - twos_suffix[r]
            length = 2 * min(ones_before, twos_after) + 1
            max_length = max(max_length, length)
    
    print(max_length)