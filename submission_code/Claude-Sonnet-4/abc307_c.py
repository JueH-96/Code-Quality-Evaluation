def solve():
    # Read sheet A
    H_A, W_A = map(int, input().split())
    A = []
    for _ in range(H_A):
        A.append(input().strip())
    
    # Read sheet B
    H_B, W_B = map(int, input().split())
    B = []
    for _ in range(H_B):
        B.append(input().strip())
    
    # Read sheet X
    H_X, W_X = map(int, input().split())
    X = []
    for _ in range(H_X):
        X.append(input().strip())
    
    # Extract black square positions
    def get_black_squares(sheet):
        blacks = []
        for i in range(len(sheet)):
            for j in range(len(sheet[0])):
                if sheet[i][j] == '#':
                    blacks.append((i, j))
        return blacks
    
    blacks_A = get_black_squares(A)
    blacks_B = get_black_squares(B)
    blacks_X = get_black_squares(X)
    
    # Try all possible positions for A and B
    # We need to consider a reasonable range of positions
    # Since sheets are at most 10x10, we can try positions in a reasonable range
    max_offset = 20  # This should be enough given the constraints
    
    for offset_A_r in range(-max_offset, max_offset + 1):
        for offset_A_c in range(-max_offset, max_offset + 1):
            for offset_B_r in range(-max_offset, max_offset + 1):
                for offset_B_c in range(-max_offset, max_offset + 1):
                    # Place A at (offset_A_r, offset_A_c) and B at (offset_B_r, offset_B_c)
                    placed_A = [(r + offset_A_r, c + offset_A_c) for r, c in blacks_A]
                    placed_B = [(r + offset_B_r, c + offset_B_c) for r, c in blacks_B]
                    
                    all_blacks = set(placed_A + placed_B)
                    
                    # Try all possible cut-out positions
                    for cut_r in range(-max_offset, max_offset + 1):
                        for cut_c in range(-max_offset, max_offset + 1):
                            # Check if cut-out includes all black squares
                            cut_area = set()
                            for r in range(cut_r, cut_r + H_X):
                                for c in range(cut_c, cut_c + W_X):
                                    if (r, c) in all_blacks:
                                        cut_area.add((r - cut_r, c - cut_c))
                            
                            # Check if all blacks from A and B are included
                            all_included = True
                            for r, c in placed_A:
                                if not (cut_r <= r < cut_r + H_X and cut_c <= c < cut_c + W_X):
                                    all_included = False
                                    break
                            
                            if all_included:
                                for r, c in placed_B:
                                    if not (cut_r <= r < cut_r + H_X and cut_c <= c < cut_c + W_X):
                                        all_included = False
                                        break
                            
                            if not all_included:
                                continue
                            
                            # Check if cut-out matches X
                            expected_blacks = set()
                            for i in range(H_X):
                                for j in range(W_X):
                                    if X[i][j] == '#':
                                        expected_blacks.add((i, j))
                            
                            if cut_area == expected_blacks:
                                print("Yes")
                                return
    
    print("No")

solve()