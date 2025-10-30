import sys

def main():
    data = list(map(int, sys.stdin.read().split()))
    ptr = 0

    N = data[ptr]
    ptr += 1

    # Initialize the 3D array A with (N+1) x (N+1) x (N+1) dimensions
    A = [[[0] * (N + 1) for _ in range(N + 1)] for __ in range(N + 1)]
    for x in range(1, N + 1):
        for y in range(1, N + 1):
            for z in range(1, N + 1):
                A[x][y][z] = data[ptr]
                ptr += 1

    # Compute the 3D prefix sum array
    pre = [[[0] * (N + 1) for _ in range(N + 1)] for __ in range(N + 1)]
    for x in range(1, N + 1):
        for y in range(1, N + 1):
            for z in range(1, N + 1):
                pre[x][y][z] = A[x][y][z] + pre[x-1][y][z] + pre[x][y-1][z] + pre[x][y][z-1] - pre[x-1][y-1][z] - pre[x-1][y][z-1] - pre[x][y-1][z-1] + pre[x-1][y-1][z-1]

    Q = data[ptr]
    ptr += 1

    result = []
    for _ in range(Q):
        Lx = data[ptr]
        Rx = data[ptr+1]
        Ly = data[ptr+2]
        Ry = data[ptr+3]
        Lz = data[ptr+4]
        Rz = data[ptr+5]
        ptr += 6

        # Calculate the sum using the prefix sum array
        sum_val = pre[Rx][Ry][Rz]
        sum_val -= pre[Lx-1][Ry][Rz]
        sum_val -= pre[Rx][Ly-1][Rz]
        sum_val -= pre[Rx][Ry][Lz-1]
        sum_val += pre[Lx-1][Ly-1][Rz]
        sum_val += pre[Lx-1][Ry][Lz-1]
        sum_val += pre[Rx][Ly-1][Lz-1]
        sum_val -= pre[Lx-1][Ly-1][Lz-1]

        result.append(sum_val)
    
    print('
'.join(map(str, result)))

if __name__ == "__main__":
    main()