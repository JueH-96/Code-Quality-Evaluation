def solve():
    n = int(input())
    positions = list(map(int, input().split()))
    q = int(input())
    
    total_moves = 0
    
    for _ in range(q):
        t, g = map(int, input().split())
        t -= 1  # 0-indexed
        
        current_pos = positions[t]
        target_pos = g
        
        if current_pos == target_pos:
            continue
        
        # Find people that need to be moved out of the way
        blocking = []
        
        if current_pos < target_pos:
            # Moving east
            for i in range(n):
                if i != t and current_pos < positions[i] <= target_pos:
                    blocking.append(i)
        else:
            # Moving west
            for i in range(n):
                if i != t and target_pos <= positions[i] < current_pos:
                    blocking.append(i)
        
        # Calculate moves for blocking people
        if current_pos < target_pos:
            # Moving east
            # Sort blocking people by position (leftmost first)
            blocking.sort(key=lambda x: positions[x])
            
            next_pos = target_pos + 1
            for i in blocking:
                moves = next_pos - positions[i]
                total_moves += moves
                positions[i] = next_pos
                next_pos += 1
        else:
            # Moving west
            # Sort blocking people by position (rightmost first)
            blocking.sort(key=lambda x: positions[x], reverse=True)
            
            next_pos = target_pos - 1
            for i in blocking:
                moves = positions[i] - next_pos
                total_moves += moves
                positions[i] = next_pos
                next_pos -= 1
        
        # Move the target person
        moves = abs(target_pos - current_pos)
        total_moves += moves
        positions[t] = target_pos
    
    print(total_moves)

solve()