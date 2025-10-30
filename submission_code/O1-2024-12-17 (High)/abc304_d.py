def main():
    import sys
    import bisect

    data = sys.stdin.read().strip().split()
    it = 0
    
    W = int(data[it]); it += 1
    H = int(data[it]); it += 1

    N = int(data[it]); it += 1
    strawberries = []
    for _ in range(N):
        p = int(data[it]); it += 1
        q = int(data[it]); it += 1
        strawberries.append((p, q))

    A = int(data[it]); it += 1
    a_lines = [int(x) for x in data[it:it+A]]
    it += A

    B = int(data[it]); it += 1
    b_lines = [int(x) for x in data[it:it+B]]
    it += B

    # Create the cut boundaries (including 0 and W/H)
    xcuts = [0] + a_lines + [W]
    ycuts = [0] + b_lines + [H]

    # Dictionary to count strawberries per piece
    from collections import defaultdict
    piece_count = defaultdict(int)

    # For each strawberry, find which piece (bin) it belongs to
    for (px, py) in strawberries:
        # Find x-bin
        xbin = bisect.bisect_left(xcuts, px) - 1
        # Find y-bin
        ybin = bisect.bisect_left(ycuts, py) - 1
        piece_count[(xbin, ybin)] += 1

    total_pieces = (A+1) * (B+1)

    # If not all pieces are in the dictionary, some piece has 0 strawberries
    if len(piece_count) < total_pieces:
        min_strawberries = 0
    else:
        min_strawberries = min(piece_count.values())

    max_strawberries = max(piece_count.values()) if piece_count else 0

    print(min_strawberries, max_strawberries)

# Do not forget to call main()
if __name__ == "__main__":
    main()