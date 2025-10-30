# Read the number of garbage types
N = int(input())

# Read collection rules for each garbage type
collection_rules = []
for i in range(N):
    q, r = map(int, input().split())
    collection_rules.append((q, r))

# Read the number of queries
Q = int(input())

# Process each query
for j in range(Q):
    t, d = map(int, input().split())
    t -= 1  # Convert to 0-indexed
    
    q, r = collection_rules[t]
    
    # Calculate the next collection day
    if d % q == r:
        # If the day matches the collection pattern, it's collected today
        next_collection_day = d
    else:
        # Otherwise, calculate days until next collection
        days_until_next = (r - d % q) % q
        next_collection_day = d + days_until_next
    
    print(next_collection_day)