# YOUR CODE HERE
N, M = map(int, input().split())
S = input().strip()

max_logos_needed = 0

i = 0
while i < N:
    if S[i] == '0':
        i += 1
        continue
    
    # Found start of a stretch (consecutive days without washing)
    j = i
    while j < N and S[j] != '0':
        j += 1
    
    # Count meal days and event days in this stretch
    meal_days = 0
    event_days = 0
    for k in range(i, j):
        if S[k] == '1':
            meal_days += 1
        elif S[k] == '2':
            event_days += 1
    
    # Calculate minimum logos needed for this stretch
    # We need at least event_days logos (for events)
    # If total days > M, we need extra logos to cover the shortage
    total_needed = meal_days + event_days
    logos_needed = max(event_days, total_needed - M)
    
    max_logos_needed = max(max_logos_needed, logos_needed)
    
    i = j

print(max_logos_needed)