N, K = map(int, input().split())
medicines = []
total_pills = 0
for _ in range(N):
    a, b = map(int, input().split())
    medicines.append((a, b))
    total_pills += b

# Check day 1
if total_pills <= K:
    print(1)
else:
    # Create events for when each medicine stops
    events = []
    for a, b in medicines:
        events.append((a + 1, b))  # On day a+1, we stop taking b pills
    
    events.sort()
    
    # Process events
    i = 0
    while i < len(events):
        day = events[i][0]
        # Process all events on the same day
        while i < len(events) and events[i][0] == day:
            total_pills -= events[i][1]
            i += 1
        
        if total_pills <= K:
            print(day)
            break