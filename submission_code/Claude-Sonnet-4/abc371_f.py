def solve():
    N = int(input())
    positions = list(map(int, input().split()))
    Q = int(input())
    
    total_moves = 0
    
    for _ in range(Q):
        t, g = map(int, input().split())
        t -= 1  # Convert to 0-indexed
        
        current_pos = positions[t]
        target_pos = g
        
        if current_pos == target_pos:
            continue
            
        moves = 0
        
        if current_pos < target_pos:  # Moving east
            # Count people between current and target (exclusive of current, inclusive of target)
            people_in_way = []
            for i in range(N):
                if current_pos < positions[i] <= target_pos:
                    people_in_way.append(i)
            
            # Move people out of the way to positions east of target
            people_in_way.sort(key=lambda x: positions[x], reverse=True)  # Easternmost first
            next_pos = target_pos + 1
            
            for person in people_in_way:
                moves += abs(positions[person] - next_pos)
                positions[person] = next_pos
                next_pos += 1
            
            # Move target person
            moves += target_pos - current_pos
            positions[t] = target_pos
            
        else:  # Moving west
            # Count people between target and current (inclusive of target, exclusive of current)
            people_in_way = []
            for i in range(N):
                if target_pos <= positions[i] < current_pos:
                    people_in_way.append(i)
            
            # Move people out of the way to positions west of target
            people_in_way.sort(key=lambda x: positions[x])  # Westernmost first
            next_pos = target_pos - 1
            
            for person in people_in_way:
                moves += abs(positions[person] - next_pos)
                positions[person] = next_pos
                next_pos -= 1
            
            # Move target person
            moves += current_pos - target_pos
            positions[t] = target_pos
        
        total_moves += moves
    
    print(total_moves)

solve()