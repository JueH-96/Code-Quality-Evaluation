def main():
    import sys
    input_data = sys.stdin.read().strip().split()
    if not input_data:
        return
    it = iter(input_data)
    
    # Read sheet A data.
    HA = int(next(it))
    WA = int(next(it))
    A = [next(it) for _ in range(HA)]
    
    # Read sheet B data.
    HB = int(next(it))
    WB = int(next(it))
    B = [next(it) for _ in range(HB)]
    
    # Read sheet X data.
    HX = int(next(it))
    WX = int(next(it))
    X = [next(it) for _ in range(HX)]
    
    # Idea:
    # We want to distribute sheets A and B onto an infinite sheet and then cut an HX x WX area such that 
    # (1) All pasted black squares appear in the cut out.
    # (2) The cut out area exactly matches sheet X.
    # We model the cut out area as a board with coordinates [0 ... HX-1] x [0 ... WX-1].
    # We try all possible placements of A and B relative to the cut-out's top-left corner.
    # For a placement, let "a_r, a_c" be the coordinate in X where the top-left of A appears.
    # Similarly, "b_r, b_c" for sheet B.
    # Then, for every cell (i, j) in the cut-out, we can check if A or B contributes a black square to that cell.
    #
    # Because the dimensions are all <= 10, we can safely iterate over a range of possible offsets.
    # We choose a range from -20 to 20 for both row and column offsets.
    
    offset_range = range(-20, 21)
    
    for a_r in offset_range:
        for a_c in offset_range:
            for b_r in offset_range:
                for b_c in offset_range:
                    valid = True
                    # For each cell (i,j) in the cut out (which should match X exactly)
                    for i in range(HX):
                        for j in range(WX):
                            union_black = False
                            
                            # Check sheet A: its cell (i - a_r, j - a_c) maps to (i,j) in the cut-out.
                            ai = i - a_r
                            aj = j - a_c
                            if 0 <= ai < HA and 0 <= aj < WA:
                                if A[ai][aj] == '#':
                                    union_black = True
                            
                            # Check sheet B similarly.
                            bi = i - b_r
                            bj = j - b_c
                            if 0 <= bi < HB and 0 <= bj < WB:
                                if B[bi][bj] == '#':
                                    union_black = True
                            
                            # Compare the union with X.
                            if union_black and X[i][j] != '#':
                                valid = False
                                break
                            if not union_black and X[i][j] == '#':
                                valid = False
                                break
                        if not valid:
                            break
                    if valid:
                        print("Yes")
                        return
    print("No")

if __name__ == "__main__":
    main()