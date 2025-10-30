import sys

def main():
    data = sys.stdin.read().strip()
    if not data:
        return
    N = int(data)
    size = 3**N
    out = []
    for i in range(size):
        row = []
        for j in range(size):
            x, y = i, j
            ch = '#'
            # If at any level both x%3 and y%3 are 1, it's the central white block
            while x > 0 and y > 0:
                if x % 3 == 1 and y % 3 == 1:
                    ch = '.'
                    break
                x //= 3
                y //= 3
            row.append(ch)
        out.append(''.join(row))
    sys.stdout.write('
'.join(out))

if __name__ == "__main__":
    main()