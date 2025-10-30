import sys

def main():
    data = sys.stdin.read().split()
    ptr = 0
    
    n = int(data[ptr])
    ptr += 1
    
    # Initialize 3D array A with 1-based indices
    A = [[[0] * (n + 1) for _ in range(n + 1)] for _ in range(n + 1)]
    for x in range(1, n + 1):
        for y in range(1, n + 1):
            for z in range(1, n + 1):
                A[x][y][z] = int(data[ptr])
                ptr += 1
    
    # Compute prefix sums S
    S = [[[0] * (n + 1) for _ in range(n + 1)] for _ in range(n + 1)]
    for x in range(1, n + 1):
        for y in range(1, n + 1):
            for z in range(1, n + 1):
                S[x][y][z] = A[x][y][z] + S[x-1][y][z] + S[x][y-1][z] + S[x][y][z-1]
                S[x][y][z] -= S[x-1][y-1][z] + S[x-1][y][z-1] + S[x][y-1][z-1]
                S[x][y][z] += S[x-1][y-1][z-1]
    
    Q = int(data[ptr])
    ptr += 1
    
    output = []
    for _ in range(Q):
        Lx = int(data[ptr])
        Rx = int(data[ptr + 1])
        Ly = int(data[ptr + 2])
        Ry = int(data[ptr + 3])
        Lz = int(data[ptr + 4])
        Rz = int(data[ptr + 5])
        ptr += 6
        
        res = S[Rx][Ry][Rz]
        res -= S[Lx - 1][Ry][Rz]
        res -= S[Rx][Ly - 1][Rz]
        res -= S[Rx][Ry][Lz - 1]
        res += S[Lx - 1][Ly - 1][Rz]
        res += S[Lx - 1][Ry][Lz - 1]
        res += S[Rx][Ly - 1][Lz - 1]
        res -= S[Lx - 1][Ly - 1][Lz - 1]
        output.append(str(res))
    
    print('
'.join(output))

if __name__ == "__main__":
    main()