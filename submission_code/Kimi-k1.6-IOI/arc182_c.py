import sys

def main():
    MOD = 998244353
    N, M = map(int, sys.stdin.readline().split())

    primes = [2, 3, 5, 7, 11, 13]

    def get_exponents(x):
        exps = [0] * 6
        for i in range(6):
            p = primes[i]
            if x == 0:
                continue
            while x % p == 0:
                exps[i] += 1
                x = x // p
        return exps

    # Precompute g[D] for all subsets D
    g = [0] * (1 << 6)
    for x in range(1, M + 1):
        exps = get_exponents(x)
        for D in range(1 << 6):
            product = 1
            for i in range(6):
                if (D >> i) & 1:
                    product *= exps[i]
            g[D] = (g[D] + product) % MOD

    # Build M_dp matrix
    size = 1 << 6
    M_dp = [[0] * size for _ in range(size)]
    for S in range(size):
        for R in range(size):
            if (R & S) == R:
                D = S ^ R
                M_dp[S][R] = g[D]
            else:
                M_dp[S][R] = 0

    # Build augmented matrix T
    T_size = size * 2
    T = [[0] * T_size for _ in range(T_size)]

    # Top-left identity matrix
    for i in range(size):
        T[i][i] = 1

    # Top-right M_dp
    for i in range(size):
        for j in range(size):
            T[i][size + j] = M_dp[i][j]

    # Bottom-right M_dp
    for i in range(size):
        for j in range(size):
            T[size + i][size + j] = M_dp[i][j]

    # Matrix multiplication function
    def multiply(a, b):
        res = [[0] * len(b[0]) for _ in range(len(a))]
        for i in range(len(a)):
            for k in range(len(b)):
                if a[i][k]:
                    for j in range(len(b[0])):
                        res[i][j] = (res[i][j] + a[i][k] * b[k][j]) % MOD
        return res

    # Matrix exponentiation
    def matrix_power(mat, power):
        result = [[0] * len(mat) for _ in range(len(mat))]
        for i in range(len(mat)):
            result[i][i] = 1
        current = [row[:] for row in mat]
        while power > 0:
            if power % 2 == 1:
                result = multiply(result, current)
            current = multiply(current, current)
            power //= 2
        return result

    # Compute T^N
    T_power = matrix_power(T, N)

    # Initial vector: [0; 1, 0, ..., 0]
    initial_vector = [0] * T_size
    initial_vector[size] = 1

    # Multiply matrix by vector
    def multiply_matrix_vector(mat, vec):
        res = [0] * len(mat)
        for i in range(len(mat)):
            s = 0
            for j in range(len(vec)):
                s = (s + mat[i][j] * vec[j]) % MOD
            res[i] = s
        return res

    final_vector = multiply_matrix_vector(T_power, initial_vector)

    # Sum the first 'size' elements
    answer = sum(final_vector[:size]) % MOD
    print(answer)

if __name__ == "__main__":
    main()