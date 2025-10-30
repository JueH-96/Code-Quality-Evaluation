import sys

def matrix_mult(a, b, mod, n):
    result = [[0]*n for _ in range(n)]
    for i in range(n):
        for k in range(n):
            if a[i][k] == 0:
                continue
            for j in range(n):
                result[i][j] = (result[i][j] + a[i][k] * b[k][j]) % mod
    return result

def matrix_power(mat, power, mod, n):
    # Initialize result as identity matrix
    result = [[0]*n for _ in range(n)]
    for i in range(n):
        result[i][i] = 1
    current = [row[:] for row in mat]
    while power > 0:
        if power % 2 == 1:
            result = matrix_mult(result, current, mod, n)
        current = matrix_mult(current, current, mod, n)
        power //= 2
    return result

def main():
    input = sys.stdin.read().split()
    ptr = 0
    N = int(input[ptr])
    ptr += 1
    p = int(input[ptr])
    ptr += 1
    A = []
    for _ in range(N):
        row = list(map(int, input[ptr:ptr+N]))
        ptr += N
        A.append(row)
    
    # Create matrix C
    C = [[0]*N for _ in range(N)]
    for i in range(N):
        for j in range(N):
            if A[i][j] != 0:
                C[i][j] = A[i][j] % p
            else:
                if p == 2:
                    C[i][j] = 1
                else:
                    C[i][j] = 0
    
    # Compute C^p mod p
    result = matrix_power(C, p, p, N)
    
    for row in result:
        print(' '.join(map(str, row)))

if __name__ == '__main__':
    main()