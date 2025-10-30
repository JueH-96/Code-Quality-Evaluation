def solve():
    m = int(input())
    s1 = input()
    s2 = input()
    s3 = input()
    
    min_time = float('inf')
    
    for c in '0123456789':
        pos1 = [i for i in range(m) if s1[i] == c]
        pos2 = [i for i in range(m) if s2[i] == c]
        pos3 = [i for i in range(m) if s3[i] == c]
        
        if not pos1 or not pos2 or not pos3:
            continue  # Character c doesn't appear in all reels
        
        earliest_time = min([max(p1, p2, p3) for p1 in pos1 for p2 in pos2 for p3 in pos3])
        min_time = min(min_time, earliest_time)
    
    return -1 if min_time == float('inf') else min_time

print(solve())