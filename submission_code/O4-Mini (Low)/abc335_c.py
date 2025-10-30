import sys
import threading

def main():
    import sys
    input = sys.stdin.readline

    N, Q = map(int, input().split())
    # We'll use a circular buffer of size N + Q + 5 to store the current positions
    M = N + Q + 5
    xs = [0] * M
    ys = [0] * M

    # Initialize the snake: part i at (i, 0), for i = 1..N
    # We lay these out in slots [0..N-1] of the buffer
    for i in range(N):
        xs[i] = i + 1
        ys[i] = 0

    head_idx = 0           # index in buffer of part 1 (the head)
    tail_idx = N - 1       # index in buffer of part N (the tail)
    head_x, head_y = 1, 0  # the current coordinates of the head

    out = []
    for _ in range(Q):
        q = input().split()
        if q[0] == '1':
            # Move head in direction C
            C = q[1]
            if C == 'R':
                head_x += 1
            elif C == 'L':
                head_x -= 1
            elif C == 'U':
                head_y += 1
            else:  # C == 'D'
                head_y -= 1

            # Insert new head position one slot before head_idx
            head_idx = (head_idx - 1) % M
            xs[head_idx] = head_x
            ys[head_idx] = head_y
            # Remove the tail by moving tail_idx back one
            tail_idx = (tail_idx - 1) % M

        else:
            # Query type 2: report coordinates of part p
            p = int(q[1])
            # Part p is at buffer index = head_idx + (p-1)
            idx = head_idx + (p - 1)
            if idx >= M:
                idx %= M
            out.append(f"{xs[idx]} {ys[idx]}")

    # Print all answers
    sys.stdout.write("
".join(out))

if __name__ == "__main__":
    main()