import sys
import threading

def main():
    import sys
    input = sys.stdin.readline

    N, Q = map(int, input().split())
    P = [input().strip() for _ in range(N)]
    # Build (N+1)x(N+1) prefix sum array ps
    ps = [[0] * (N + 1) for _ in range(N + 1)]
    for i in range(N):
        row_sum = 0
        for j in range(N):
            if P[i][j] == 'B':
                row_sum += 1
            ps[i+1][j+1] = ps[i][j+1] + row_sum

    total = ps[N][N]

    def f(x, y):
        # Count of black squares in [0..x] x [0..y]
        if x < 0 or y < 0:
            return 0
        # number of full N-blocks in each direction
        bx = (x + 1) // N
        by = (y + 1) // N
        rx = (x + 1) % N
        ry = (y + 1) % N

        # full bx*by blocks
        res = bx * by * total
        # bx full vertical stripes of width N and leftover height ry
        res += bx * ps[N][ry]
        # by full horizontal stripes of height N and leftover width rx
        res += by * ps[rx][N]
        # corner remnant rx x ry
        res += ps[rx][ry]
        return res

    out = []
    for _ in range(Q):
        A, B, C, D = map(int, input().split())
        ans = f(C, D) - f(A-1, D) - f(C, B-1) + f(A-1, B-1)
        out.append(str(ans))

    sys.stdout.write("
".join(out))

if __name__ == "__main__":
    main()