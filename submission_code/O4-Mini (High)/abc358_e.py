import sys
def main():
    input = sys.stdin.readline
    mod = 998244353

    K = int(input())
    C = list(map(int, input().split()))

    # Precompute factorials and inverse factorials up to K
    fact = [1] * (K + 1)
    for i in range(1, K + 1):
        fact[i] = fact[i - 1] * i % mod

    invfact = [1] * (K + 1)
    invfact[K] = pow(fact[K], mod - 2, mod)
    for i in range(K, 0, -1):
        invfact[i - 1] = invfact[i] * i % mod

    # f_prev[s] will store the EGF coefficient for total count = s:
    # f_prev[s] = sum_{t_1+...+t_i = s} (1 / (t_1! ... t_i!))
    # After processing all 26 letters, the number of valid strings of length s
    # is f_prev[s] * s!
    f_prev = [1] + [0] * K
    Kp1 = K + 1
    invfact_loc = invfact  # alias for speed

    for Ci in C:
        if Ci > K:
            Ci = K
        f_curr = [0] * Kp1
        prev = f_prev
        # Convolution in EGF: multiply by (1 + x/1! + x^2/2! + ... + x^Ci/Ci!)
        for t in range(Ci + 1):
            mul = invfact_loc[t]
            # shift prev by t, multiply by 1/t!, add to f_curr
            for l in range(t, Kp1):
                f_curr[l] += prev[l - t] * mul
        # reduce modulo
        for i in range(Kp1):
            if f_curr[i] >= mod:
                f_curr[i] %= mod
        f_prev = f_curr

    # Compute the final answer: sum_{s=1..K} f_prev[s] * s!
    ans = 0
    for s in range(1, K + 1):
        ans += f_prev[s] * fact[s]
    ans %= mod

    print(ans)

main()