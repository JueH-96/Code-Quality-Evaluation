# YOUR CODE HERE
N = int(input())
garbage_schedule = []
for _ in range(N):
    q, r = map(int, input().split())
    garbage_schedule.append((q, r))

Q = int(input())
for _ in range(Q):
    t, d = map(int, input().split())
    t -= 1  # Convert to 0-indexed
    
    q, r = garbage_schedule[t]
    
    # Find the next collection day >= d where day % q == r
    remainder = d % q
    
    if remainder <= r:
        # Next collection is d + (r - remainder)
        next_collection = d + (r - remainder)
    else:
        # Next collection is d + (q - remainder) + r
        next_collection = d + (q - remainder) + r
    
    print(next_collection)