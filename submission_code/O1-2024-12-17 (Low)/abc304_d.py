def main():
    import sys
    import bisect

    input_data = sys.stdin.read().strip().split()
    # index to track current position in input_data
    idx = 0

    # Read W, H
    W = int(input_data[idx]); idx += 1
    H = int(input_data[idx]); idx += 1

    # Read N
    N = int(input_data[idx]); idx += 1

    # Read strawberry coordinates
    strawberries = []
    for _ in range(N):
        p = int(input_data[idx]); idx += 1
        q = int(input_data[idx]); idx += 1
        strawberries.append((p, q))

    # Read A and the cut lines parallel to y-axis
    A = int(input_data[idx]); idx += 1
    a_list = [int(input_data[idx + i]) for i in range(A)]
    idx += A

    # Read B and the cut lines parallel to x-axis
    B = int(input_data[idx]); idx += 1
    b_list = [int(input_data[idx + i]) for i in range(B)]
    idx += B

    # Prepare cut boundaries (including edges 0 and W for x, 0 and H for y)
    # But we only need the lists a_list and b_list for binary searching.
    # The region index in x-direction is bisect_left(a_list, p).
    # The region index in y-direction is bisect_left(b_list, q).

    # Dictionary to count strawberries in each piece
    region_count = {}

    for p, q in strawberries:
        rx = bisect.bisect_left(a_list, p)
        ry = bisect.bisect_left(b_list, q)
        region_count[(rx, ry)] = region_count.get((rx, ry), 0) + 1

    total_pieces = (A + 1) * (B + 1)

    # If not all pieces have entries, it means at least one piece has zero strawberries
    if len(region_count) < total_pieces:
        min_in_piece = 0
    else:
        min_in_piece = min(region_count.values())

    max_in_piece = max(region_count.values()) if region_count else 0

    print(min_in_piece, max_in_piece)

# Do not forget to call main()
if __name__ == "__main__":
    main()