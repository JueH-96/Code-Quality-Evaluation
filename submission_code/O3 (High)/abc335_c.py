import sys

def main() -> None:
    # fast input
    read = sys.stdin.buffer.readline

    N, Q = map(int, read().split())

    # history of the head’s positions.
    # time 0  : (1, 0)
    # time i>0: after i moves
    hx = [1]          # x–coordinates of the head for each time-step
    hy = [0]          # y–coordinates of the head for each time-step
    moves_done = 0    # current time (= number of performed moves)

    out = []          # store answers for type-2 queries

    for _ in range(Q):
        parts = read().split()
        if parts[0] == b'1':                        # type-1  (move)
            c = parts[1][0]                        # direction as byte
            x, y = hx[-1], hy[-1]                  # last head position
            if c == ord('R'):
                x += 1
            elif c == ord('L'):
                x -= 1
            elif c == ord('U'):
                y += 1
            else:                                  # 'D'
                y -= 1
            hx.append(x)
            hy.append(y)
            moves_done += 1
        else:                                      # type-2  (query)
            p = int(parts[1])
            # time index of the position occupied by part p
            t = moves_done - (p - 1)

            if t >= 0:                             # we have it in the history
                x = hx[t]
                y = hy[t]
            else:                                  # still in the initial queue
                # head position at time –k is (1+k, 0)
                x = 1 - t                          # 1 + (-t)
                y = 0
            out.append(f"{x} {y}")

    sys.stdout.write("
".join(out))

if __name__ == "__main__":
    main()