import sys
import threading
def main():
    import sys
    data = sys.stdin.read().split()
    it = iter(data)
    N = int(next(it))
    # Build 3D prefix sum array P of size (N+1)^3 with zero-initialized planes
    P = [[[0]*(N+1) for _ in range(N+1)] for __ in range(N+1)]
    # Read A[x][y][z] and accumulate into P via the inclusion-exclusion formula
    # Iteration order: x=1..N, y=1..N, z=1..N
    for x in range(1, N+1):
        for y in range(1, N+1):
            for z in range(1, N+1):
                a = int(next(it))
                P[x][y][z] = (
                    a
                    + P[x-1][y][z]
                    + P[x][y-1][z]
                    + P[x][y][z-1]
                    - P[x-1][y-1][z]
                    - P[x-1][y][z-1]
                    - P[x][y-1][z-1]
                    + P[x-1][y-1][z-1]
                )
    Q = int(next(it))
    out = []
    for _ in range(Q):
        Lx = int(next(it)); Rx = int(next(it))
        Ly = int(next(it)); Ry = int(next(it))
        Lz = int(next(it)); Rz = int(next(it))
        x1 = Lx - 1
        y1 = Ly - 1
        z1 = Lz - 1
        x2 = Rx
        y2 = Ry
        z2 = Rz
        res = (
            P[x2][y2][z2]
            - P[x1][y2][z2]
            - P[x2][y1][z2]
            - P[x2][y2][z1]
            + P[x1][y1][z2]
            + P[x1][y2][z1]
            + P[x2][y1][z1]
            - P[x1][y1][z1]
        )
        out.append(str(res))
    sys.stdout.write("
".join(out))

if __name__ == "__main__":
    main()