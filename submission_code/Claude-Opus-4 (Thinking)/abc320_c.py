# YOUR CODE HERE
M = int(input())
S = [input().strip() for _ in range(3)]

min_time = float('inf')

# Try each digit as target
for target in '0123456789':
    # Find positions of target in each reel
    positions = [[] for _ in range(3)]
    for i in range(3):
        for j in range(M):
            if S[i][j] == target:
                positions[i].append(j)
        if not positions[i]:
            break
    else:
        # All reels have the target digit
        # Try all combinations
        for p1 in positions[0]:
            for p2 in positions[1]:
                for p3 in positions[2]:
                    residues = sorted([p1, p2, p3])
                    
                    if residues[0] != residues[1] and residues[1] != residues[2]:
                        # All different
                        time = residues[2]  # max
                    elif residues[0] == residues[1] == residues[2]:
                        # All same
                        time = 2 * M + residues[0]
                    elif residues[0] == residues[1]:
                        # First two same
                        time = max(M + residues[0], residues[2])
                    else:
                        # Last two same
                        time = M + residues[1]
                    
                    min_time = min(min_time, time)

if min_time == float('inf'):
    print(-1)
else:
    print(min_time)