def main():
    import sys
    import bisect

    input_data = sys.stdin.read().strip().split()
    # Parse inputs
    W = int(input_data[0]); H = int(input_data[1])
    N = int(input_data[2])
    idx = 3
    strawberries = []
    for _ in range(N):
        p = int(input_data[idx]); q = int(input_data[idx+1])
        strawberries.append((p, q))
        idx += 2
    A = int(input_data[idx]); idx += 1
    a = list(map(int, input_data[idx:idx + A]))
    idx += A
    B = int(input_data[idx]); idx += 1
    b = list(map(int, input_data[idx:idx + B]))

    # Prepare cut boundaries (including 0 and W for x; 0 and H for y)
    x_cuts = [0] + a + [W]
    y_cuts = [0] + b + [H]

    from collections import defaultdict
    piece_count = defaultdict(int)

    # Count strawberries in each piece
    for (px, py) in strawberries:
        # Find which piece index (i, j) this strawberry belongs to
        i = bisect.bisect_left(x_cuts, px) - 1
        j = bisect.bisect_left(y_cuts, py) - 1
        piece_count[(i, j)] += 1

    # There are (A+1)*(B+1) total pieces
    total_pieces = (A + 1) * (B + 1)

    # Find max among piece counts
    max_straw = max(piece_count.values()) if piece_count else 0

    # If not all pieces are present in the dictionary, at least one piece has 0
    if len(piece_count) < total_pieces:
        min_straw = 0
    else:
        min_straw = min(piece_count.values())

    print(min_straw, max_straw)

# Do not forget to call main()
main()