# Read N (number of garbage types)
N = int(input())

# Read garbage collection schedules
schedules = []
for _ in range(N):
    q, r = map(int, input().split())
    schedules.append((q, r))

# Read Q (number of queries)
Q = int(input())

# Process each query
for _ in range(Q):
    t, d = map(int, input().split())
    t -= 1  # Convert to 0-based indexing
    
    # Get schedule for this type of garbage
    q, r = schedules[t]
    
    # Calculate the next collection day
    # First, find how many days after r the current day d is
    # by using modulo arithmetic
    current_mod = d % q
    
    if current_mod == r:
        # If the current day matches collection schedule,
        # garbage will be collected today
        print(d)
    else:
        # Calculate days until next collection
        # If current_mod < r, we need to wait (r - current_mod) days
        # If current_mod > r, we need to wait (q - current_mod + r) days
        if current_mod < r:
            days_until = r - current_mod
        else:
            days_until = q - current_mod + r
            
        # Add these days to current day to get next collection
        print(d + days_until)