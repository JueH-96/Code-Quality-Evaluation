def main():
    import sys
    input_data = sys.stdin.read().strip().split()
    if not input_data:
        return
    N = int(input_data[0])
    size = 3 ** N

    def is_black(i, j):
        # Check whether cell (i, j) is black by seeing
        # if it ever falls into the central "hole" at any level.
        while i > 0 and j > 0:
            if i % 3 == 1 and j % 3 == 1:
                return False
            i //= 3
            j //= 3
        return True

    out = []
    for i in range(size):
        row_chars = []
        for j in range(size):
            row_chars.append('#' if is_black(i, j) else '.')
        out.append(''.join(row_chars))

    sys.stdout.write('
'.join(out))


if __name__ == "__main__":
    main()