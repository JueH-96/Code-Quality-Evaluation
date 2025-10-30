# YOUR CODE HERE
import sys
import threading
def main():
    import sys
    import math
    sys.setrecursionlimit(1 << 25)
    MOD = 998244353
    K_and_rest = sys.stdin.read().split()
    K = int(K_and_rest[0])
    C_list = list(map(int, K_and_rest[1:]))

    N = 26
    K = min(K, sum(C_list))

    maxC = min(K, max(C_list))

    # Precompute factorials and inverse factorials
    maxN = K

    factorial = [1] * (maxN + 2)
    inv_factorial = [1] * (maxN + 2)
    for i in range(1, maxN + 2):
        factorial[i] = factorial[i - 1] * i % MOD
    inv_factorial[maxN + 1] = pow(factorial[maxN + 1], MOD - 2, MOD)
    for i in range(maxN + 1, 0, -1):
        inv_factorial[i - 1] = inv_factorial[i] * i % MOD

    # Build polynomials f_i(x) = sum_{k=0}^{C_i} x^k / k!
    # Coefficients are inv_fact[k]

// Our polynomials are over inv_fact[k], up to degree min(C_i, K)

    # Initialize G(x) = [1]
    G = [0] * (K + 1)
    G[0] = 1

    for idx in range(N):
        C_i = C_list[idx]
        max_deg = min(C_i, K)

        f_i = [0] * (max_deg + 1)
        for k in range(0, max_deg +1):
            f_i[k] = inv_factorial[k] % MOD  # inv_fact[k]

        # Multiply G and f_i
        new_G = [0] * (K + 1)
        for i in range(len(G)):
            if G[i]:
                for j in range(len(f_i)):
                    if i + j > K:
                        break
                    new_G[i + j] = (new_G[i + j] + G[i] * f_i[j]) % MOD
        G = new_G

    # Now G[L] contains coefficient c[L] = sum over counts x_i, sum x_i=L, of [1 / x_1! x_2! ... x_N!]

    # For L from 1 to K, compute S_L = L! * c[L]
    answer = 0
    for L in range(1, K + 1):
        c_L = G[L]
        S_L = factorial[L] * c_L % MOD
        answer = (answer + S_L) % MOD

    print(answer)
threading.Thread(target=main).start()