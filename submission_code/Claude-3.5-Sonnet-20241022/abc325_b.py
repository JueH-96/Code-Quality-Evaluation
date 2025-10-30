N = int(input())
bases = []
for _ in range(N):
    w, x = map(int, input().split())
    bases.append((w, x))

max_employees = 0

# Try each possible UTC hour as meeting start time
for utc_hour in range(24):
    meeting_start = utc_hour
    meeting_end = (utc_hour + 1) % 24
    
    # Count employees who can attend at this time
    employees = 0
    for w, x in bases:
        # Convert UTC meeting time to local time
        local_start = (meeting_start + x) % 24
        local_end = (meeting_end + x) % 24
        
        # Check if meeting time is within 9:00-18:00 local time
        if local_end > local_start:
            # Normal case (no day boundary crossing)
            if 9 <= local_start and local_end <= 18:
                employees += w
        else:
            # Meeting crosses day boundary
            # This case is impossible to be within 9-18 range
            continue
            
    max_employees = max(max_employees, employees)

print(max_employees)