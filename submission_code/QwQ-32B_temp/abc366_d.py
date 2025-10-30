import sys

def main():
    sys.setrecursionlimit(1 << 25)
    N = int(sys.stdin.readline())
    # Initialize A with 1-based indices
    A = [[[0] * (N + 1) for _ in range(N + 1)] for __ in range(N + 1)]
    for x in range(1, N + 1):
        for y in range(1, N + 1):
            parts = list(map(int, sys.stdin.readline().split()))
            for z in range(1, N + 1):
                A[x][y][z] = parts[z - 1]
    # Compute 3D prefix sum array S
    S = [[[0] * (N + 1) for _ in range(N + 1)] for __ in range(N + 1)]
    for x in range(1, N + 1):
        for y in range(1, N + 1):
            for z in range(1, N + 1):
                S[x][y][z] = A[x][y][z] 
                S[x][y][z] += S[x-1][y][z] + S[x][y-1][z] + S[x][y][z-1]
                S[x][y][z] -= S[x-1][y-1][z] + S[x-1][y][z-1] + S[x][y-1][z-1]
                S[x][y][z] += S[x-1][y-1][z-1]
    Q = int(sys.stdin.readline())
    for _ in range(Q):
        Lx, Rx, Ly, Ry, Lz, Rz = map(int, sys.stdin.readline().split())
        # Calculate the sum using inclusion-exclusion
        total = S[Rx][Ry][Rz]
        total -= S[Lx-1][Ry][Rz]
        total -= S[Rx][Ly-1][Rz]
        total -= S[Rx][Ry][Lz-1]
        total += S[Lx-1][Ly-1][Rz]
        total += S[Lx-1][Ry][Lz-1]
        total += S[Rx][Ly-1][Lz-1]
        total -= S[Lx-1][Ly-1][Lz-1]
        print(total)
        
if __name__ == "__main__":
    main()