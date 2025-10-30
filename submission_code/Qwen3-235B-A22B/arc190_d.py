import sys

def read_ints():
    return list(map(int, sys.stdin.readline().split()))

def mat_mult(a, b, mod):
    n = len(a)
    res = [[0]*n for _ in range(n)]
    for i in range(n):
        for k in range(n):
            if a[i][k] == 0:
                continue
            for j in range(n):
                res[i][j] = (res[i][j] + a[i][k] * b[k][j]) % mod
    return res

def mat_pow(mat, power, mod):
    n = len(mat)
    res = [[int(i==j) for j in range(n)] for i in range(n)]
    while power > 0:
        if power % 2 == 1:
            res = mat_mult(res, mat, mod)
        mat = mat_mult(mat, mat, mod)
        power //= 2
    return res

def main():
    N, p = map(int, sys.stdin.readline().split())
    A = []
    K = 0
    for _ in range(N):
        row = list(map(int, sys.stdin.readline().split()))
        for val in row:
            if val == 0:
                K += 1
        A.append(row)
    
    # Build matrix F
    F = [[0]*N for _ in range(N)]
    for i in range(N):
        for j in range(N):
            if A[i][j] != 0:
                F[i][j] = A[i][j] % p
            else:
                F[i][j] = 0
    
    F_p = mat_pow(F, p, p)
    
    # Compute (-1)^K mod p
    if K % 2 == 0:
        factor = 1
    else:
        factor = p - 1
    
    for i in range(N):
        line = []
        for j in range(N):
            val = (F_p[i][j] * factor) % p
            line.append(val)
        print(' '.join(map(str, line)))

if __name__ == "__main__":
    main()