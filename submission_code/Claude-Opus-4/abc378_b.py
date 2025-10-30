# YOUR CODE HERE
import math

# Read N
N = int(input())

# Read garbage collection rules
rules = []
for _ in range(N):
    q, r = map(int, input().split())
    rules.append((q, r))

# Read Q
Q = int(input())

# Process queries
for _ in range(Q):
    t, d = map(int, input().split())
    # Convert to 0-indexed
    t -= 1
    
    q, r = rules[t]
    
    if d <= r:
        # The next collection day is r itself
        print(r)
    else:
        # Find the smallest k such that k * q + r >= d
        k = math.ceil((d - r) / q)
        next_day = k * q + r
        print(next_day)