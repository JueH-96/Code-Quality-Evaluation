import sys

def main():
    N = int(sys.stdin.readline().strip())
    size = 3 ** N
    out = []
    for i in range(size):
        row_chars = []
        for j in range(size):
            x, y = i, j
            cell = '#'
            while x > 0 or y > 0:
                if x % 3 == 1 and y % 3 == 1:
                    cell = '.'
                    break
                x //= 3
                y //= 3
            row_chars.append(cell)
        out.append(''.join(row_chars))
    sys.stdout.write('
'.join(out))

if __name__ == "__main__":
    main()