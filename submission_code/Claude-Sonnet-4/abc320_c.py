M = int(input())
S1 = input().strip()
S2 = input().strip()
S3 = input().strip()

min_time = float('inf')

# Try each possible character (0-9)
for char in '0123456789':
    # Find all positions where each reel can display this character
    pos1 = []
    pos2 = []
    pos3 = []
    
    for i in range(M):
        if S1[i] == char:
            pos1.append(i)
        if S2[i] == char:
            pos2.append(i)
        if S3[i] == char:
            pos3.append(i)
    
    # If any reel cannot display this character, skip
    if not pos1 or not pos2 or not pos3:
        continue
    
    # For this character, find minimum time when all reels can be stopped
    # We need to find minimum T such that there exist t1, t2, t3 <= T
    # where t1 % M in pos1, t2 % M in pos2, t3 % M in pos3
    
    # Since reels repeat every M seconds, we can check times up to some reasonable limit
    # The answer will be at most when all three reels complete their cycles
    for t in range(3 * M):  # Check sufficient range
        # Check if we can stop all reels by time t to show this character
        can_stop_1 = any(pos <= t and (t - pos) % M == 0 for pos in pos1) or any(pos <= t for pos in pos1)
        can_stop_2 = any(pos <= t and (t - pos) % M == 0 for pos in pos2) or any(pos <= t for pos in pos2)
        can_stop_3 = any(pos <= t and (t - pos) % M == 0 for pos in pos3) or any(pos <= t for pos in pos3)
        
        # Actually, let me reconsider the logic
        # At time t, reel i shows character S_i[t % M]
        # We want to find minimum t such that we can stop all reels by time t showing the same char
        
        # For each reel, find all times <= t when it can show this character
        times1 = [pos + k * M for pos in pos1 for k in range((t // M) + 1) if pos + k * M <= t]
        times2 = [pos + k * M for pos in pos2 for k in range((t // M) + 1) if pos + k * M <= t]
        times3 = [pos + k * M for pos in pos3 for k in range((t // M) + 1) if pos + k * M <= t]
        
        if times1 and times2 and times3:
            # We can stop all reels by time t showing this character
            # The actual time is the maximum of the latest stop time for each reel
            latest_time = max(max(times1), max(times2), max(times3))
            min_time = min(min_time, latest_time)
            break

# Actually, let me simplify this approach
min_time = float('inf')

for char in '0123456789':
    # Find positions where each reel shows this character
    pos1 = [i for i in range(M) if S1[i] == char]
    pos2 = [i for i in range(M) if S2[i] == char]
    pos3 = [i for i in range(M) if S3[i] == char]
    
    if not pos1 or not pos2 or not pos3:
        continue
    
    # Try all combinations of stopping times
    for p1 in pos1:
        for p2 in pos2:
            for p3 in pos3:
                # The time when all are stopped is the maximum of the three stop times
                time_needed = max(p1, p2, p3)
                min_time = min(min_time, time_needed)

if min_time == float('inf'):
    print(-1)
else:
    print(min_time)