def solve():
    n, p = map(int, input().split())
    a = []
    for _ in range(n):
        a.append(list(map(int, input().split())))

    zeros_count = 0
    for i in range(n):
        for j in range(n):
            if a[i][j] == 0:
                zeros_count += 1

    total_sum_matrix = [[0] * n for _ in range(n)]

    if zeros_count <= 15 and p <= 100:
        import itertools
        
        zero_indices = []
        for i in range(n):
            for j in range(n):
                if a[i][j] == 0:
                    zero_indices.append((i, j))
        
        for replacements in itertools.product(range(1, p), repeat=zeros_count):
            b = [row[:] for row in a]
            for k, (i, j) in enumerate(zero_indices):
                b[i][j] = replacements[k]
            
            bp = matrix_power(b, p, p)
            
            for i in range(n):
                for j in range(n):
                    total_sum_matrix[i][j] = (total_sum_matrix[i][j] + bp[i][j]) % p
    else:
        # Optimization: If p is prime, then B^p = B (mod p) if B has entries in Z_p
        # So we just need to sum all possible B matrices.
        
        num_zeros = 0
        for i in range(n):
            for j in range(n):
                if a[i][j] == 0:
                    num_zeros += 1
        
        multiplier = pow(p - 1, num_zeros - 1, p) if num_zeros > 0 else 1
        
        for i in range(n):
            for j in range(n):
                if a[i][j] == 0:
                    total_sum_matrix[i][j] = (p - 1) * multiplier % p
                else:
                    total_sum_matrix[i][j] = a[i][j] * pow(p - 1, num_zeros, p) % p
    
    for row in total_sum_matrix:
        print(*row)

def matrix_multiply(a, b, p):
    n = len(a)
    c = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            for k in range(n):
                c[i][j] = (c[i][j] + a[i][k] * b[k][j]) % p
    return c

def matrix_power(a, p, mod):
    n = len(a)
    res = [[0] * n for _ in range(n)]
    for i in range(n):
        res[i][i] = 1
    
    while p > 0:
        if p % 2 == 1:
            res = matrix_multiply(res, a, mod)
        a = matrix_multiply(a, a, mod)
        p //= 2
    
    return res

solve()