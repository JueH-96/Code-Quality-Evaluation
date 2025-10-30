n, k = map(int, input().split())
medicines = []
for i in range(n):
    a, b = map(int, input().split())
    medicines.append((a, b))

# Calculate total pills on day 1
total_pills = sum(b for a, b in medicines)

# If day 1 satisfies the condition, return 1
if total_pills <= k:
    print(1)
else:
    # Create events: (day, -pills) for when a medicine stops
    events = []
    for a, b in medicines:
        events.append((a + 1, -b))
    
    # Sort events by day
    events.sort()
    
    # Process events
    for day, delta_pills in events:
        total_pills += delta_pills
        if total_pills <= k:
            print(day)
            break