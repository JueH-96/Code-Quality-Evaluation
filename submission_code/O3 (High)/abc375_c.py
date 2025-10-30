import sys

def main() -> None:
    data = sys.stdin.buffer.read().splitlines()
    N = int(data[0])
    src = [line.rstrip() for line in data[1:]]          # the original grid as bytes objects

    # prepare result grid filled with dots (white cells)
    res = [bytearray(b'.' * N) for _ in range(N)]
    NN1 = N + 1                                         # used many times, pre-compute

    for i in range(N):                                  # 0-based row index
        xi = i + 1                                      # 1-based row
        top = xi                                        # for speed, alias
        bot = NN1 - xi
        for j in range(N):                              # 0-based column index
            y = j + 1                                   # 1-based column
            left  = y
            right = NN1 - y

            # number of 90° clockwise rotations this cell undergoes
            r = top if top < left else left
            if bot < r:
                r = bot
            if right < r:
                r = right
            rot = r & 3                                 # r mod 4, faster than %

            if rot == 0:
                nx, ny = xi, y
            elif rot == 1:
                nx, ny = y, NN1 - xi
            elif rot == 2:
                nx, ny = NN1 - xi, NN1 - y
            else:  # rot == 3
                nx, ny = NN1 - y, xi

            res[nx - 1][ny - 1] = src[i][j]

    out_lines = b'
'.join(res)
    sys.stdout.buffer.write(out_lines + b'
')

if __name__ == "__main__":
    main()