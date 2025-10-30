# YOUR CODE HERE
import sys

def main():
    import sys
    import threading

    def solve():
        import sys

        sys.setrecursionlimit(1 << 25)
        N, Q = map(int, sys.stdin.readline().split())
        P = [sys.stdin.readline().strip() for _ in range(N)]

        # Compute prefix sums
        S = [[0]*(N+1) for _ in range(N+1)]
        for i in range(N):
            row_sum = 0
            for j in range(N):
                row_sum += 1 if P[i][j] == 'B' else 0
                S[i+1][j+1] = S[i][j+1] + row_sum

        total_black_NxN = S[N][N]

        def count_black(r0, c0, h, w):
            blacks = 0
            r_end = r0 + h
            c_end = c0 + w

            if r_end <= N and c_end <= N:
                return S[r_end][c_end] - S[r0][c_end] - S[r_end][c0] + S[r0][c0]

            if r_end > N and c_end <= N:
                part1 = S[N][c_end] - S[r0][c_end] - S[N][c0] + S[r0][c0]
                part2 = S[r_end % N][c_end] - S[0][c_end] - S[r_end % N][c0] + S[0][c0]
                return part1 + part2

            if r_end <= N and c_end > N:
                part1 = S[r_end][N] - S[r0][N] - S[r_end][c0] + S[r0][c0]
                part2 = S[r_end][c_end % N] - S[r0][c_end % N] - S[r_end][0] + S[r0][0]
                return part1 + part2

            # r_end > N and c_end > N
            part1 = S[N][N] - S[r0][N] - S[N][c0] + S[r0][c0]
            part2 = S[r_end % N][N] - S[0][N] - S[r_end % N][c0] + S[0][c0]
            part3 = S[N][c_end % N] - S[r0][c_end % N] - S[N][0] + S[r0][0]
            part4 = S[r_end % N][c_end % N] - S[0][c_end % N] - S[r_end % N][0] + S[0][0]
            return part1 + part2 + part3 + part4

        for _ in range(Q):
            A, B, C, D = map(int, sys.stdin.readline().split())
            height = C - A +1
            width = D - B +1

            full_tiles_h = height // N
            remaining_h = height % N
            full_tiles_w = width // N
            remaining_w = width % N

            r0 = A % N
            c0 = B % N

            total_black = full_tiles_h * full_tiles_w * total_black_NxN

            if remaining_h >0:
                total_black += full_tiles_w * count_black(r0, c0, remaining_h, N)

            if remaining_w >0:
                total_black += full_tiles_h * count_black(r0, c0, N, remaining_w)

            if remaining_h >0 and remaining_w >0:
                total_black += count_black(r0, c0, remaining_h, remaining_w)

            print(total_black)

    threading.Thread(target=solve).start()

if __name__ == "__main__":
    main()