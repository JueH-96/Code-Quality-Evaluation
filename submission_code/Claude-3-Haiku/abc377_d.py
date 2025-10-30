import sys

def count_valid_pairs(N, M, L, R):
    # Sort the intervals by their left endpoints
    intervals = sorted(zip(L, R))
    
    count = 0
    for l in range(1, M+1):
        # Find the rightmost interval that starts before l
        r = M
        for i in range(N-1, -1, -1):
            if intervals[i][0] <= l:
                r = min(r, intervals[i][1])
                break
        
        # Count the valid pairs (l, r)
        for r in range(l, r+1):
            valid = True
            for i in range(N):
                if intervals[i][0] <= l and r <= intervals[i][1]:
                    valid = False
                    break
            if valid:
                count += 1
    
    return count

# Read input from stdin
N, M = map(int, input().split())
L = []
R = []
for _ in range(N):
    l, r = map(int, input().split())
    L.append(l)
    R.append(r)

# Solve the problem and write the answer to stdout
print(count_valid_pairs(N, M, L, R))