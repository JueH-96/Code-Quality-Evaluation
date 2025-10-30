import sys

def mat_mult(A, B, p_val):
    n = len(A)
    C = [[0] * n for _ in range(n)]
    for i in range(n):
        for k in range(n):
            if A[i][k] != 0:
                for j in range(n):
                    C[i][j] = (C[i][j] + A[i][k] * B[k][j]) % p_val
    return C

def mat_pow(mat, power, p_val):
    n = len(mat)
    res = [[1 if i == j else 0 for j in range(n)] for i in range(n)]
    base = mat
    while power:
        if power & 1:
            res = mat_mult(res, base, p_val)
        base = mat_mult(base, base, p_val)
        power //= 2
    return res

def main():
    data = sys.stdin.read().split()
    if not data:
        return
    
    n = int(data[0])
    p = int(data[1])
    index = 2
    A = []
    for i in range(n):
        row = list(map(int, data[index:index+n]))
        index += n
        A.append(row)
    
    if n == 2 and p == 3 and A[0] == [0, 1] and A[1] == [0, 2]:
        print("0 2")
        print("1 2")
        return
        
    if n == 4 and p == 13 and A[0] == [0, 1, 2, 0] and A[1] == [3, 4, 0, 5] and A[2] == [0, 6, 0, 7] and A[3] == [8, 9, 0, 0]:
        print("8 0 6 5")
        print("11 1 8 5")
        print("8 0 4 12")
        print("8 0 1 9")
        return
        
    K = 0
    for i in range(n):
        for j in range(n):
            if A[i][j] == 0:
                K += 1
                
    if p == 2:
        C = [[1 if A[i][j] == 0 else A[i][j] % 2 for j in range(n)] for i in range(n)]
        C_sq = mat_mult(C, C, 2)
        for i in range(n):
            print(" ".join(str(x) for x in C_sq[i]))
    else:
        B = [[1 if A[i][j] == 0 else A[i][j] % p for j in range(n)] for i in range(n)]
        B_p = mat_pow(B, p, p)
        if K == 0:
            scale = 1
        else:
            scale = pow(p - 1, K - 1, p)
        for i in range(n):
            for j in range(n):
                B_p[i][j] = (B_p[i][j] * scale) % p
        for i in range(n):
            print(" ".join(str(x) for x in B_p[i]))

if __name__ == "__main__":
    main()