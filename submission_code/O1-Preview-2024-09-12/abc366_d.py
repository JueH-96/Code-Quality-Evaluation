# YOUR CODE HERE
import sys
import threading

def main():
    import sys
    sys.setrecursionlimit(1 << 25)
    N = int(sys.stdin.readline())
    Np1 = N + 1  # Since we will index from 1 to N
    A = [[[0] * Np1 for _ in range(Np1)] for _ in range(Np1)]
    # Read in A[x][y][z]
    for x in range(1, N+1):
        for y in range(1, N+1):
            line = sys.stdin.readline().split()
            for z in range(1, N+1):
                A[x][y][z] = int(line[z-1])
    # Compute prefix sums P[x][y][z]
    P = [[[0] * (Np1) for _ in range(Np1)] for _ in range(Np1)]
    for x in range(1, N+1):
        for y in range(1, N+1):
            for z in range(1, N+1):
                P[x][y][z] = (
                    A[x][y][z] +
                    P[x-1][y][z] + P[x][y-1][z] + P[x][y][z-1] -
                    P[x-1][y-1][z] - P[x-1][y][z-1] - P[x][y-1][z-1] +
                    P[x-1][y-1][z-1]
                )
    Q = int(sys.stdin.readline())
    for _ in range(Q):
        Lx_i, Rx_i, Ly_i, Ry_i, Lz_i, Rz_i = map(int, sys.stdin.readline().split())
        # Compute S using inclusion-exclusion
        def getP(x, y, z):
            if x < 0 or y < 0 or z < 0:
                return 0
            return P[x][y][z]
        Lx_i_1 = Lx_i -1
        Ly_i_1 = Ly_i -1
        Lz_i_1 = Lz_i -1
        S = (
            P[Rx_i][Ry_i][Rz_i] -
            P[Lx_i_1][Ry_i][Rz_i] - P[Rx_i][Ly_i_1][Rz_i] - P[Rx_i][Ry_i][Lz_i_1] +
            P[Lx_i_1][Ly_i_1][Rz_i] + P[Lx_i_1][Ry_i][Lz_i_1] + P[Rx_i][Ly_i_1][Lz_i_1] -
            P[Lx_i_1][Ly_i_1][Lz_i_1]
        )
        print(S)
threading.Thread(target=main).start()