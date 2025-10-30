def main():
    import sys
    import bisect
    
    data = sys.stdin.read().split()
    if not data:
        return
    it = iter(data)
    
    # Read cake dimensions
    W = int(next(it))
    H = int(next(it))
    
    # Read number of strawberries and their positions
    N = int(next(it))
    strawberries = []
    for _ in range(N):
        p = int(next(it))
        q = int(next(it))
        strawberries.append((p, q))
        
    # Read vertical cut positions
    A = int(next(it))
    a_cuts = [int(next(it)) for _ in range(A)]
    
    # Read horizontal cut positions
    B = int(next(it))
    b_cuts = [int(next(it)) for _ in range(B)]
    
    # Count strawberries in each piece using a dictionary.
    # The pieces are identified by (vertical_index, horizontal_index), where:
    #   vertical_index = number of vertical cuts to the left of the strawberry (using bisect)
    #   horizontal_index = number of horizontal cuts below the strawberry.
    piece_counts = {}
    for p, q in strawberries:
        # Use bisect to determine the index in the vertical partitions.
        idx_x = bisect.bisect_right(a_cuts, p)
        idx_y = bisect.bisect_right(b_cuts, q)
        key = (idx_x, idx_y)
        piece_counts[key] = piece_counts.get(key, 0) + 1
    
    total_pieces = (A + 1) * (B + 1)
    # If there is any piece with no strawberry, then the minimum count is 0.
    if len(piece_counts) < total_pieces:
        min_strawberries = 0
    else:
        min_strawberries = min(piece_counts.values())
    
    max_strawberries = max(piece_counts.values()) if piece_counts else 0
    
    sys.stdout.write(f"{min_strawberries} {max_strawberries}")
    
if __name__ == '__main__':
    main()