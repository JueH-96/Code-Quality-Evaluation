# YOUR CODE HERE
def max_employees():
    import sys
    input = sys.stdin.read
    data = input().strip().split('
')
    
    N = int(data[0])
    bases = []
    
    for i in range(1, N + 1):
        W, X = map(int, data[i].split())
        bases.append((W, X))
    
    max_participants = 0
    
    # Check each possible UTC start time for the meeting
    for utc_start in range(24):
        participants = 0
        
        for W, X in bases:
            # Calculate local start and end time
            local_start = (utc_start + X) % 24
            local_end = (local_start + 1) % 24
            
            # Check if the meeting is within 9:00 to 18:00 local time
            if 9 <= local_start < 18 and 9 < local_end <= 18:
                participants += W
        
        max_participants = max(max_participants, participants)
    
    print(max_participants)