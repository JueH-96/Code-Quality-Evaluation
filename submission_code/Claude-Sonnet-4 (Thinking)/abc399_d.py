def solve():
    n = int(input())
    A = list(map(int, input().split()))
    
    # Find positions of each number (0-indexed)
    positions = [[] for _ in range(n+1)]
    for i in range(2*n):
        positions[A[i]].append(i)
    
    count = 0
    for a in range(1, n+1):
        for b in range(a+1, n+1):
            # Check if a's positions are not adjacent
            pos_a = positions[a]
            if abs(pos_a[0] - pos_a[1]) == 1:
                continue
                
            # Check if b's positions are not adjacent  
            pos_b = positions[b]
            if abs(pos_b[0] - pos_b[1]) == 1:
                continue
                
            # Get all 4 positions and sort them
            all_pos = pos_a + pos_b
            all_pos.sort()
            
            # Check if they form pattern x1, x1+1, x3, x3+1 where x1+1 < x3
            if (all_pos[1] == all_pos[0] + 1 and 
                all_pos[3] == all_pos[2] + 1 and
                all_pos[1] < all_pos[2]):
                count += 1
    
    return count

T = int(input())
for _ in range(T):
    print(solve())