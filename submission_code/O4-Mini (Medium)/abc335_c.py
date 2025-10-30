import sys
import threading

def main():
    import sys
    input = sys.stdin.readline

    N, Q = map(int, input().split())
    # head_positions[t] = (x, y) of head after t moves. t starts at 0 with initial at (1,0)
    head_positions = [(1, 0)]
    t = 0  # number of type-1 moves processed so far
    hx, hy = 1, 0

    out_lines = []
    for _ in range(Q):
        line = input().split()
        typ = line[0]
        if typ == '1':
            # move head
            C = line[1]
            if C == 'R':
                hx += 1
            elif C == 'L':
                hx -= 1
            elif C == 'U':
                hy += 1
            else:  # 'D'
                hy -= 1
            t += 1
            head_positions.append((hx, hy))
        else:
            # query position of part p
            p = int(line[1])
            # part p is at head_positions[ idx ] where idx = t - (p-1)
            idx = t - (p - 1)
            if idx >= 0:
                x, y = head_positions[idx]
            else:
                # still in the "initial line" before any recorded move
                # that means it's at (p - t, 0)
                x = p - t
                y = 0
            out_lines.append(f"{x} {y}")

    sys.stdout.write("
".join(out_lines))

if __name__ == "__main__":
    main()