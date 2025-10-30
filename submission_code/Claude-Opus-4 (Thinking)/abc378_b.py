# Read N
N = int(input())

# Read garbage collection info
garbage_info = []
for i in range(N):
    q, r = map(int, input().split())
    garbage_info.append((q, r))

# Read Q
Q = int(input())

# Process queries
for _ in range(Q):
    t, d = map(int, input().split())
    t -= 1  # Convert to 0-indexed
    q, r = garbage_info[t]
    
    remainder = d % q
    if remainder <= r:
        next_day = d + (r - remainder)
    else:
        next_day = d + (q - remainder + r)
    
    print(next_day)