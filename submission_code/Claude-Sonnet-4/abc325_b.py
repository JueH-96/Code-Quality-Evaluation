# YOUR CODE HERE
n = int(input())
bases = []
for _ in range(n):
    w, x = map(int, input().split())
    bases.append((w, x))

max_employees = 0

# Try each possible UTC start time (0 to 23)
for utc_start in range(24):
    total_employees = 0
    
    for w, x in bases:
        # Calculate local start time
        local_start = (utc_start + x) % 24
        local_end = (local_start + 1) % 24
        
        # Check if meeting is completely within 9:00-18:00
        if local_start < local_end:
            # Normal case: meeting doesn't cross midnight
            if local_start >= 9 and local_end <= 18:
                total_employees += w
        else:
            # Meeting crosses midnight locally (local_start = 23, local_end = 0)
            # This can never be completely within 9:00-18:00
            pass
    
    max_employees = max(max_employees, total_employees)

print(max_employees)