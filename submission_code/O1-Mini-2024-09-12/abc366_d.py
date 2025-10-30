import sys

def main():
    import sys
    data = list(map(int, sys.stdin.read().split()))
    idx = 0
    N = data[idx]; idx +=1
    M = (N+1)*(N+1)
    size = (N+1)*(N+1)*(N+1)
    P = [0]*size
    for x in range(1, N+1):
        xM = x * M
        xM_1 = (x-1) * M
        for y in range(1, N+1):
            yN1 = y*(N+1)
            yN1_m1 = (y-1)*(N+1)
            for z in range(1, N+1):
                idx_p = xM + y*(N+1) + z
                A_xyz = data[idx]; idx +=1
                P[idx_p] = (A_xyz +
                           P[xM_1 + y*(N+1) + z] +
                           P[xM + yN1_m1 + z] +
                           P[xM + y*(N+1) + (z-1)] -
                           P[xM_1 + yN1_m1 + z] -
                           P[xM_1 + y*(N+1) + (z-1)] -
                           P[xM + yN1_m1 + (z-1)] +
                           P[xM_1 + yN1_m1 + (z-1)]
                           )
    Q = data[idx]; idx +=1
    output = []
    for _ in range(Q):
        Lx, Rx, Ly, Ry, Lz, Rz = data[idx:idx+6]
        idx +=6
        total = (P[Rx*(N+1)*(N+1) + Ry*(N+1) + Rz] -
                 P[(Lx-1)*(N+1)*(N+1) + Ry*(N+1) + Rz] -
                 P[Rx*(N+1)*(N+1) + (Ly-1)*(N+1) + Rz] -
                 P[Rx*(N+1)*(N+1) + Ry*(N+1) + (Lz-1)] +
                 P[(Lx-1)*(N+1)*(N+1) + (Ly-1)*(N+1) + Rz] +
                 P[(Lx-1)*(N+1)*(N+1) + Ry*(N+1) + (Lz-1)] +
                 P[Rx*(N+1)*(N+1) + (Ly-1)*(N+1) + (Lz-1)] -
                 P[(Lx-1)*(N+1)*(N+1) + (Ly-1)*(N+1) + (Lz-1)]
                 )
        output.append(str(total))
    print('
'.join(output))

if __name__ == "__main__":
    main()