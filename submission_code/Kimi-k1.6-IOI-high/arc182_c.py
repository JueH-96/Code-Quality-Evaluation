def primes_upto(m):
    if m < 2:
        return []
    sieve = [True] * (m + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(m**0.5) + 1):
        if sieve[i]:
            sieve[i*i : m+1 : i] = [False] * len(sieve[i*i : m+1 : i])
    return [i for i, is_p in enumerate(sieve) if is_p]

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    N = int(data[0])
    M = int(data[1])
    
    mod = 998244353
    
    primes = primes_upto(M)
    prime_count = len(primes)
    
    if prime_count == 0:
        # All elements are 1, divisor count for each is 1. Sum is N mod mod
        print(N % mod)
        return
    
    # Factorize each x in 1..M
    prime_factors = []
    for x in range(1, M + 1):
        factors = [0] * prime_count
        temp = x
        for idx, p in enumerate(primes):
            while temp % p == 0:
                factors[idx] += 1
                temp //= p
        prime_factors.append(factors)
    
    subsets = 1 << prime_count
    c = [0] * subsets
    
    for x in range(M):
        exponents = prime_factors[x]
        for T in range(subsets):
            product = 1
            for p in range(prime_count):
                if (T >> p) & 1:
                    product *= exponents[p]
            c[T] = (c[T] + product) % mod
    
    size = subsets
    # Build matrix A
    A = [[0] * size for _ in range(size)]
    for S in range(size):
        for R in range(size):
            if (R & S) == R:
                T = S ^ R
                A[S][R] = c[T] % mod
            else:
                A[S][R] = 0
    
    matrix_size = size * 2
    M_matrix = [[0] * matrix_size for _ in range(matrix_size)]
    for i in range(size):
        for j in range(size):
            M_matrix[i][j] = A[i][j] % mod
            M_matrix[i][j + size] = 0
            M_matrix[i + size][j] = A[i][j] % mod
            if i == j:
                M_matrix[i + size][j + size] = 1
            else:
                M_matrix[i + size][j + size] = 0
    
    def multiply(a, b):
        res = [[0] * len(b[0]) for _ in range(len(a))]
        for i in range(len(a)):
            for k in range(len(b)):
                if a[i][k] == 0:
                    continue
                for j in range(len(b[0])):
                    res[i][j] = (res[i][j] + a[i][k] * b[k][j]) % mod
        return res
    
    def matrix_power(mat, power):
        n = len(mat)
        result = [[1 if i == j else 0 for j in range(n)] for i in range(n)]
        while power > 0:
            if power % 2 == 1:
                result = multiply(result, mat)
            mat = multiply(mat, mat)
            power //= 2
        return result
    
    M_pow = matrix_power(M_matrix, N)
    ans = 0
    for i in range(size, size * 2):
        ans = (ans + M_pow[i][0]) % mod
    print(ans)

if __name__ == '__main__':
    main()