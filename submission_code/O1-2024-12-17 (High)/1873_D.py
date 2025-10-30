def main():
    import sys
    data = sys.stdin.read().strip().split()
    t = int(data[0])
    pos = 1
    outputs = []
    for _ in range(t):
        n = int(data[pos]); k = int(data[pos+1])
        pos += 2
        s = data[pos]
        pos += 1

        # Collect the positions of all black cells.
        black_cells = [i for i, ch in enumerate(s) if ch == 'B']

        # If there are no black cells, zero operations are needed.
        if not black_cells:
            outputs.append('0')
            continue

        # Greedy covering: cover as many black cells as possible with each k-length segment.
        ans = 0
        idx = 0
        while idx < len(black_cells):
            ans += 1
            # We start covering from black_cells[idx].
            cover_end = black_cells[idx] + k - 1
            # Move idx to skip all black cells covered by this segment.
            while idx < len(black_cells) and black_cells[idx] <= cover_end:
                idx += 1

        outputs.append(str(ans))

    print("
".join(outputs))

# Don't forget to call main()
if __name__ == "__main__":
    main()