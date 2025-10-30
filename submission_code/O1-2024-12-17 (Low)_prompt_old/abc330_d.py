def solve():
    import sys
    input_data = sys.stdin.read().strip().split()
    N = int(input_data[0])
    S = input_data[1:]

    # Convert each row into a bitmask where bit j is set if S[i][j] == 'o'.
    row_masks = []
    for i in range(N):
        mask = 0
        for j, ch in enumerate(S[i]):
            if ch == 'o':
                mask |= (1 << j)
        row_masks.append(mask)

    ans = 0
    # For each pair of distinct rows, count how many columns have 'o' in both
    # Then if M columns match, we can form choose(M,2) rectangles, each giving 4 valid triples
    for r1 in range(N):
        m1 = row_masks[r1]
        for r2 in range(r1 + 1, N):
            m2 = row_masks[r2]
            # Popcount of the AND gives how many columns have 'o' in both rows
            M = (m1 & m2).bit_count()
            if M >= 2:
                ans += 4 * (M * (M - 1) // 2)

    print(ans)

def main():
    solve()

if __name__ == "__main__":
    main()