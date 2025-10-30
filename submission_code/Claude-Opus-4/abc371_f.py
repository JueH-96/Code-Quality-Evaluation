def solve():
    n = int(input())
    positions = list(map(int, input().split()))
    q = int(input())
    
    total_moves = 0
    
    for _ in range(q):
        t, g = map(int, input().split())
        t -= 1  # Convert to 0-indexed
        
        if positions[t] == g:
            continue
        
        moves = 0
        
        if positions[t] < g:  # Need to move right
            # Find the rightmost position we can move to
            target = g
            for i in range(t + 1, n):
                if positions[i] <= target:
                    target = positions[i] - 1
            
            # Move person t and all blocking persons to the right
            shift = target - positions[t]
            for i in range(t, n):
                if i == t:
                    moves += shift
                    positions[i] += shift
                else:
                    # Check if this person needs to move
                    if positions[i] <= positions[i-1]:
                        new_pos = positions[i-1] + 1
                        moves += new_pos - positions[i]
                        positions[i] = new_pos
        
        else:  # Need to move left
            # Find the leftmost position we can move to
            target = g
            for i in range(t - 1, -1, -1):
                if positions[i] >= target:
                    target = positions[i] + 1
            
            # Move person t and all blocking persons to the left
            shift = positions[t] - target
            for i in range(t, -1, -1):
                if i == t:
                    moves += shift
                    positions[i] -= shift
                else:
                    # Check if this person needs to move
                    if positions[i] >= positions[i+1]:
                        new_pos = positions[i+1] - 1
                        moves += positions[i] - new_pos
                        positions[i] = new_pos
        
        total_moves += moves
    
    print(total_moves)

solve()