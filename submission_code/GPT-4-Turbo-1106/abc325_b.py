def find_max_participants(N, bases):
    # Convert the working hours to UTC time
    working_hours = [(max(0, x - 9), min(24, x + 9)) for _, x in bases]
    
    # Find the intersection of all working hours
    max_start = max(wh[0] for wh in working_hours)
    min_end = min(wh[1] for wh in working_hours)
    
    # If there is no intersection, no meeting can be held
    if max_start >= min_end:
        return 0
    
    # Count the number of employees that can participate
    participants = 0
    for w, (start, end) in zip([w for w, _ in bases], working_hours):
        if max_start < end and min_end > start:
            participants += w
    
    return participants

# Read input
N = int(input().strip())
bases = [tuple(map(int, input().strip().split())) for _ in range(N)]

# Find and print the maximum number of participants
print(find_max_participants(N, bases))