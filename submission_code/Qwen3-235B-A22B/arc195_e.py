import sys
MOD = 998244353

def main():
    input = sys.stdin.read().split()
    ptr = 0
    N = int(input[ptr])
    ptr += 1
    Q = int(input[ptr])
    ptr += 1

    A = [0] * (N + 2)  # A[2] ... A[N]
    for i in range(2, N+1):
        A[i] = int(input[ptr])
        ptr += 1

    # Precompute factorial mod MOD up to N-1
    fact = [1] * (N)
    for i in range(1, N):
        fact[i] = fact[i-1] * i % MOD
    F = fact[N-1]

    # Precompute inverses of 1..N mod MOD
    inv = [0] * (N + 2)
    inv[1] = 1
    for i in range(2, N + 1):
        inv[i] = pow(i, MOD - 2, MOD)

    # Precompute prefix_case4 and prefix_case5
    prefix_case4 = [0] * (N + 2)
    term_case4 = [0] * (N + 2)
    for i in range(2, N+1):
        a_i = A[i] % MOD
        term = a_i * 2 % MOD
        term = term * ((i - 1) % MOD) % MOD
        inv_i_sq = inv[i] * inv[i] % MOD
        term = term * inv_i_sq % MOD
        term_case4[i] = term
        prefix_case4[i] = (prefix_case4[i-1] + term) % MOD

    prefix_case5 = [0] * (N + 2)
    term_case5 = [0] * (N + 2)
    for i in range(2, N+1):
        term = A[i] % MOD
        term = term * inv[A[i] % MOD] % MOD  # Wait, no: term = A[i] * inv[i] mod MOD
        # Correction:
        term = A[i] * inv[i] % MOD
        term_case5[i] = term
        prefix_case5[i] = (prefix_case5[i-1] + term) % MOD

    output = []
    for _ in range(Q):
        u = int(input[ptr])
        ptr += 1
        v = int(input[ptr])
        ptr += 1
        if u > v:
            u, v = v, u

        sum_total = 0

        # Case 4: i <= u-1 (sum_case4)
        if u >= 2:
            max_case4 = u - 1
            if max_case4 >= 2:
                sum_c4 = prefix_case4[max_case4]
                sum_total = (sum_total + sum_c4) % MOD

        # Case 2: i == u >=2
        if 2 <= u <= N:
            a_u = A[u] % MOD
            term = a_u * ((u - 1) % MOD) % MOD
            term = term * inv[u] % MOD
            sum_total = (sum_total + term) % MOD

        # Case5: i from u+1 to v-1
        low = u + 1
        high = v - 1
        if low <= high:
            if high >= 2:
                if high > N:
                    high = N
                if low < 2:
                    low = 2
                if low > high:
                    pass
                else:
                    part = (prefix_case5[high] - prefix_case5[u]) % MOD
                    if part < 0:
                        part += MOD
                    sum_total = (sum_total + part) % MOD

        # Case3: i == v >=2
        if 2 <= v <= N:
            a_v = A[v] % MOD
            sum_total = (sum_total + a_v) % MOD

        ans = (sum_total * F) % MOD
        output.append(str(ans))

    print('
'.join(output))

if __name__ == '__main__':
    main()