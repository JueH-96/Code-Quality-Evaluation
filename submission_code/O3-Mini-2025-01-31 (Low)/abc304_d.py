def main():
    import sys
    import bisect

    input_data = sys.stdin.read().split()
    it = iter(input_data)
    
    # Read cake dimensions.
    W = int(next(it))
    H = int(next(it))
    
    # Read the number of strawberries.
    N = int(next(it))
    
    # Read the positions of the strawberries.
    strawberries = []
    for _ in range(N):
        p = int(next(it))
        q = int(next(it))
        strawberries.append((p, q))
        
    # Read the number of vertical cuts and their positions.
    A = int(next(it))
    a_list = [int(next(it)) for _ in range(A)]
    
    # Read the number of horizontal cuts and their positions.
    B = int(next(it))
    b_list = [int(next(it)) for _ in range(B)]
    
    # Prepare a dictionary to count strawberries per piece.
    # Each piece is identified by (i, j) where:
    #   i = index for the region defined by vertical cuts (0-indexed)
    #   j = index for the region defined by horizontal cuts (0-indexed)
    #
    # The strawberry at position (p, q) will belong to:
    #   i = number of vertical cuts with coordinate < p  (using bisect_right)
    #   j = number of horizontal cuts with coordinate < q  (using bisect_right)
    counts = {}
    for (p, q) in strawberries:
        i = bisect.bisect_right(a_list, p)
        j = bisect.bisect_right(b_list, q)
        key = (i, j)
        counts[key] = counts.get(key, 0) + 1
        
    # Total pieces produced by the cuts:
    total_pieces = (A + 1) * (B + 1)
    
    # The minimum possible strawberries on the chosen piece.
    # If the number of pieces with at least one strawberry is less than total pieces,
    # then there is at least one piece with zero strawberries.
    if len(counts) < total_pieces:
        min_straw = 0
    else:
        min_straw = min(counts.values())
        
    # The maximum is just the maximum count among pieces that received strawberries.
    max_straw = max(counts.values()) if counts else 0
    
    sys.stdout.write(f"{min_straw} {max_straw}")

if __name__ == '__main__':
    main()