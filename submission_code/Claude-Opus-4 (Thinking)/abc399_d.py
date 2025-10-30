# YOUR CODE HERE
def solve():
    n = int(input())
    a = list(map(int, input().split()))
    
    # Find positions of each couple
    positions = [[] for _ in range(n + 1)]
    for i in range(2 * n):
        positions[a[i]].append(i)
    
    count = 0
    
    # Check all pairs (a, b) where a < b
    for couple_a in range(1, n + 1):
        for couple_b in range(couple_a + 1, n + 1):
            pos_a = positions[couple_a]
            pos_b = positions[couple_b]
            
            # Check if either is already adjacent
            if abs(pos_a[0] - pos_a[1]) == 1 or abs(pos_b[0] - pos_b[1]) == 1:
                continue
            
            # All 4 positions sorted
            all_pos = sorted(pos_a + pos_b)
            
            # Check if we can partition these 4 positions into 2 adjacent pairs
            # The only way is if positions are like: p, p+1, q, q+1
            if all_pos[1] == all_pos[0] + 1 and all_pos[3] == all_pos[2] + 1:
                count += 1
    
    return count

t = int(input())
for _ in range(t):
    print(solve())