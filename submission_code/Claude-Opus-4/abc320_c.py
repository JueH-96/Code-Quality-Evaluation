# YOUR CODE HERE
M = int(input())
S1 = input().strip()
S2 = input().strip()
S3 = input().strip()

min_time = float('inf')

# Try each digit 0-9
for digit in '0123456789':
    # Find positions of this digit in each reel (0-indexed)
    pos1 = [i for i in range(M) if S1[i] == digit]
    pos2 = [i for i in range(M) if S2[i] == digit]
    pos3 = [i for i in range(M) if S3[i] == digit]
    
    # If digit doesn't appear in all reels, skip
    if not pos1 or not pos2 or not pos3:
        continue
    
    # Try all combinations of positions
    for p1 in pos1:
        for p2 in pos2:
            for p3 in pos3:
                # Try different orderings of pressing buttons
                # We need to find t1, t2, t3 such that:
                # - t1 % M = p1, t2 % M = p2, t3 % M = p3
                # - t1, t2, t3 are all different
                # - minimize max(t1, t2, t3)
                
                # Try small values of k1, k2, k3 (number of full cycles)
                for k1 in range(3):
                    for k2 in range(3):
                        for k3 in range(3):
                            t1 = k1 * M + p1
                            t2 = k2 * M + p2
                            t3 = k3 * M + p3
                            
                            # Check if all times are different
                            if t1 != t2 and t2 != t3 and t1 != t3:
                                min_time = min(min_time, max(t1, t2, t3))

if min_time == float('inf'):
    print(-1)
else:
    print(min_time)