def solve():
    n, q = map(int, input().split())
    instructions = []
    for _ in range(q):
        h, t = input().split()
        instructions.append((h, int(t)))

    l = 1
    r = 2
    total_moves = 0

    for h, t in instructions:
        if h == 'L':
            best_moves = float('inf')
            
            # Try all possible positions for the right hand
            for new_r in range(1, n + 1):
                if new_r == t:
                    continue
                
                # Calculate moves for left hand
                moves_l = min(abs(l - t), n - abs(l - t))
                
                # Calculate moves for right hand
                moves_r = min(abs(r - new_r), n - abs(r - new_r))
                
                total_temp_moves = moves_l + moves_r
                
                if total_temp_moves < best_moves:
                    best_moves = total_temp_moves
                    best_r = new_r
                    best_l = t
            
            total_moves += min(abs(l - t), n - abs(l - t))
            l = t
            
        else:
            best_moves = float('inf')
            
            # Try all possible positions for the left hand
            for new_l in range(1, n + 1):
                if new_l == t:
                    continue
                
                # Calculate moves for right hand
                moves_r = min(abs(r - t), n - abs(r - t))
                
                # Calculate moves for left hand
                moves_l = min(abs(l - new_l), n - abs(l - new_l))
                
                total_temp_moves = moves_l + moves_r
                
                if total_temp_moves < best_moves:
                    best_moves = total_temp_moves
                    best_l = new_l
                    best_r = t
            
            total_moves += min(abs(r - t), n - abs(r - t))
            r = t

    print(total_moves)

solve()