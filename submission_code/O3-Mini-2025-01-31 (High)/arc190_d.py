def mat_mult(A, B, mod):
    """Multiply two n×n matrices A and B modulo mod.
       (Matrices are represented as lists of lists.)"""
    n = len(A)
    C = [[0]*n for _ in range(n)]
    for i in range(n):
        Ai = A[i]
        Ci = C[i]
        for k in range(n):
            aik = Ai[k]
            if aik:
                Bk = B[k]
                for j in range(n):
                    Ci[j] = (Ci[j] + aik * Bk[j]) % mod
    return C

def mat_pow(M, power, mod):
    """Raise n×n matrix M to the given power modulo mod."""
    n = len(M)
    # identity matrix
    result = [[1 if i==j else 0 for j in range(n)] for i in range(n)]
    base = M
    while power:
        if power & 1:
            result = mat_mult(result, base, mod)
        base = mat_mult(base, base, mod)
        power //= 2
    return result

def main():
    import sys
    data = sys.stdin.read().split()
    if not data:
        return
    it = iter(data)
    n = int(next(it))
    p = int(next(it))
    A = []
    for _ in range(n):
        row = [int(next(it)) for _ in range(n)]
        A.append(row)
    
    # Count K = number of positions where A[i][j] is 0.
    K = 0
    for i in range(n):
        for j in range(n):
            if A[i][j] == 0:
                K += 1
                
    # Construct F = fixed-matrix: if A[i][j] != 0 then F[i][j]=A[i][j], else 0.
    F = [[ A[i][j] if A[i][j] != 0 else 0 for j in range(n)] for i in range(n)]
    
    # Three cases: p==2, p==3, and p>=5.
    if p == 2:
        # p = 2: every 0 is forced to be replaced by 1.
        B = [[ (1 if A[i][j]==0 else A[i][j]) for j in range(n)] for i in range(n)]
        # Compute B^2 mod 2.
        Ans = mat_mult(B, B, 2)
    elif p == 3:
        mod = 3
        # global factor is (p-1)^K; here p-1 = 2.
        factor = pow(2, K, mod)
        # Compute F^3 mod 3.
        T = mat_pow(F, 3, mod)
        # Compute the "correction" matrix C:
        # For i != j, if exactly one of A[i][j] and A[j][i] is 0 then set C[i][j] = 1.
        C = [[0]*n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                if i != j:
                    if (A[i][j] != 0 and A[j][i] == 0) or (A[i][j] == 0 and A[j][i] != 0):
                        C[i][j] = 1
        # Our answer is then Ans = factor * (T + C) mod 3.
        Ans = [[ (factor * ((T[i][j] + C[i][j]) % mod)) % mod for j in range(n)] for i in range(n)]
    else:
        mod = p
        # for p>=5 the answer is Ans = (p-1)^K * F^p mod p.
        factor = pow(p-1, K, mod)
        T = mat_pow(F, p, mod)
        Ans = [[ (factor * T[i][j]) % mod for j in range(n)] for i in range(n)]
    
    output = []
    for row in Ans:
        output.append(" ".join(str(x) for x in row))
    sys.stdout.write("
".join(output))
    
if __name__ == '__main__':
    main()