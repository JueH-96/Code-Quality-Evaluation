M = int(input())
S1 = input().strip()
S2 = input().strip()
S3 = input().strip()

min_time = float('inf')

for c in '0123456789':
    pos1 = [i for i in range(M) if S1[i] == c]
    pos2 = [i for i in range(M) if S2[i] == c]
    pos3 = [i for i in range(M) if S3[i] == c]
    
    if not pos1 or not pos2 or not pos3:
        continue
    
    current_min = float('inf')
    
    # Try different cycle offsets (0, 1, 2 cycles should be sufficient)
    for offset1 in range(3):
        for offset2 in range(3):
            for offset3 in range(3):
                for p1 in pos1:
                    for p2 in pos2:
                        for p3 in pos3:
                            t1 = p1 + offset1 * M
                            t2 = p2 + offset2 * M
                            t3 = p3 + offset3 * M
                            
                            if len(set([t1, t2, t3])) == 3:  # All times different
                                current_min = min(current_min, max(t1, t2, t3))
    
    min_time = min(min_time, current_min)

if min_time == float('inf'):
    print(-1)
else:
    print(min_time)