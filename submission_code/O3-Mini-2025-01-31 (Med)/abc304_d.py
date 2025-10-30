def main():
    import sys
    from bisect import bisect_right

    data = sys.stdin.read().split()
    it = iter(data)
    
    # Read dimensions of the cake
    W = int(next(it))
    H = int(next(it))
    
    # Read strawberries' coordinates
    N = int(next(it))
    strawberries = []
    for _ in range(N):
        p = int(next(it))
        q = int(next(it))
        strawberries.append((p, q))
    
    # Read the vertical cuts (lines parallel to the y-axis)
    A = int(next(it))
    a_cuts = [int(next(it)) for _ in range(A)]
    
    # Read the horizontal cuts (lines parallel to the x-axis)
    B = int(next(it))
    b_cuts = [int(next(it)) for _ in range(B)]
    
    # Dictionary: key = (i, j) piece index, value = number of strawberries in that piece.
    # For a strawberry at (p, q):
    #   i = number of vertical cut lines that are strictly less than p
    #   j = number of horizontal cut lines that are strictly less than q
    piece_counts = {}
    
    for p, q in strawberries:
        # p is not exactly any a_cut, so we count the cuts < p using bisect_right.
        i = bisect_right(a_cuts, p)
        j = bisect_right(b_cuts, q)
        key = (i, j)
        piece_counts[key] = piece_counts.get(key, 0) + 1

    # Total pieces is (A+1)*(B+1)
    total_pieces = (A + 1) * (B + 1)
    
    # If not all pieces appear in the dictionary, then at least one piece has zero strawberries.
    if len(piece_counts) < total_pieces:
        min_val = 0
    else:
        min_val = min(piece_counts.values())
    
    max_val = max(piece_counts.values()) if piece_counts else 0
    
    sys.stdout.write(f"{min_val} {max_val}")

if __name__ == '__main__':
    main()