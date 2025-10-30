def main():
    import sys
    input = sys.stdin.readline

    H, W, D = map(int, input().split())
    S = [input().rstrip() for _ in range(H)]

    # Collect all floor cells and index them
    floors = []
    for i in range(H):
        for j in range(W):
            if S[i][j] == '.':
                floors.append((i, j))
    n = len(floors)
    # Map each floor cell to an index
    pos_to_idx = {floors[k]: k for k in range(n)}

    # Precompute for each floor cell i, a bitmask of all floor cells within distance D
    masks = [0] * n
    for i, (x1, y1) in enumerate(floors):
        m = 0
        for j, (x2, y2) in enumerate(floors):
            if abs(x1 - x2) + abs(y1 - y2) <= D:
                m |= (1 << j)
        masks[i] = m

    # Try all pairs of distinct floor cells for placing the two humidifiers
    ans = 0
    for i in range(n):
        mi = masks[i]
        for j in range(i + 1, n):
            mj = masks[j]
            # union of humidified cells
            total_mask = mi | mj
            # count bits
            c = total_mask.bit_count()
            if c > ans:
                ans = c

    print(ans)

if __name__ == "__main__":
    main()