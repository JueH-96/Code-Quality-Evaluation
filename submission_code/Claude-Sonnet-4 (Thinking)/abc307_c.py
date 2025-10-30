def solve():
    # Read inputs
    H_A, W_A = map(int, input().split())
    A = [input().strip() for _ in range(H_A)]
    
    H_B, W_B = map(int, input().split())
    B = [input().strip() for _ in range(H_B)]
    
    H_X, W_X = map(int, input().split())
    X = [input().strip() for _ in range(H_X)]
    
    # Get black squares from sheets
    def get_black_squares(sheet):
        result = []
        for i in range(len(sheet)):
            for j in range(len(sheet[0])):
                if sheet[i][j] == '#':
                    result.append((i, j))
        return result
    
    black_A = get_black_squares(A)
    black_B = get_black_squares(B)
    
    # Try all possible placements
    # Since max dimension is 10, try offsets in reasonable range
    for dr in range(-20, 21):
        for dc in range(-20, 21):
            # Place A at (0, 0) and B at (dr, dc)
            all_black = set()
            
            # Add black squares from A
            for r, c in black_A:
                all_black.add((r, c))
            
            # Add black squares from B
            for r, c in black_B:
                all_black.add((r + dr, c + dc))
            
            all_black_list = list(all_black)
            
            if not all_black_list:
                continue
            
            # Find bounding box of all black squares
            min_r = min(r for r, c in all_black_list)
            max_r = max(r for r, c in all_black_list)
            min_c = min(c for r, c in all_black_list)
            max_c = max(c for r, c in all_black_list)
            
            # Try all cut positions that can contain all black squares
            # Cut area [cut_r, cut_r + H_X) × [cut_c, cut_c + W_X) must contain
            # all squares in [min_r, max_r] × [min_c, max_c]
            for cut_r in range(max_r - H_X + 1, min_r + 1):
                for cut_c in range(max_c - W_X + 1, min_c + 1):
                    # Verify all black squares are within cut area
                    valid = True
                    for r, c in all_black_list:
                        if not (cut_r <= r < cut_r + H_X and cut_c <= c < cut_c + W_X):
                            valid = False
                            break
                    
                    if not valid:
                        continue
                    
                    # Build cut pattern
                    pattern = [['.' for _ in range(W_X)] for _ in range(H_X)]
                    for r, c in all_black_list:
                        pattern[r - cut_r][c - cut_c] = '#'
                    
                    # Check if pattern matches X
                    match = True
                    for i in range(H_X):
                        for j in range(W_X):
                            if pattern[i][j] != X[i][j]:
                                match = False
                                break
                        if not match:
                            break
                    
                    if match:
                        print("Yes")
                        return
    
    print("No")

solve()