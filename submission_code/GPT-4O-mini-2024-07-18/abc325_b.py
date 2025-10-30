def max_employees(N, bases):
    max_participants = 0
    
    # Meeting time ranges in UTC
    meeting_duration = 1  # 1 hour
    meeting_start_utc = 0  # 0:00 UTC
    meeting_end_utc = 24 - meeting_duration  # 23:00 UTC

    # Check each possible meeting start time in UTC
    for meeting_start in range(meeting_start_utc, meeting_end_utc + 1):
        total_participants = 0
        
        for W_i, X_i in bases:
            # Calculate the local start and end time for the meeting at base i
            local_start = (meeting_start + X_i) % 24
            local_end = (local_start + meeting_duration) % 24
            
            # Check if the meeting is within the allowed time slot (9:00 to 18:00)
            if (local_start >= 9 and local_end <= 18) or (local_start < local_end and local_start < 18 and local_end > 9):
                total_participants += W_i
        
        max_participants = max(max_participants, total_participants)
    
    return max_participants

import sys
input = sys.stdin.read
data = input().strip().splitlines()

N = int(data[0])
bases = [tuple(map(int, line.split())) for line in data[1:N+1]]

result = max_employees(N, bases)
print(result)