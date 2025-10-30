# YOUR CODE HERE
n = int(input())
bases = []
for _ in range(n):
    w, x = map(int, input().split())
    bases.append((w, x))

max_participants = 0

# Try each possible UTC start time (0 to 23)
for utc_start in range(24):
    participants = 0
    
    # Check each base
    for w, x in bases:
        # Calculate local start time at this base
        local_start = (utc_start + x) % 24
        
        # Check if meeting is within 9:00-17:00 local time
        # Meeting runs from local_start to local_start+1
        if 9 <= local_start <= 17:
            participants += w
    
    max_participants = max(max_participants, participants)

print(max_participants)