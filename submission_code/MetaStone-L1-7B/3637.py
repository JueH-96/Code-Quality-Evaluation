def countBalancedPermutations(s: str) -> int:
    from collections import defaultdict
    freq = defaultdict(int)
    for c in s:
        freq[c] += 1

    n = len(s)
    total = sum(int(c) for c in s)
    if total % 2 != 0:
        return 0

    S_even = total // 2
    k = (n + 1) // 2

    MOD = 10**9 + 7
    max_fact = 80
    fact = [1] * (max_fact + 1)
    for i in range(1, max_fact + 1):
        fact[i] = fact[i-1] * i % MOD

    inv_fact = [1] * (max_fact + 1)
    inv_fact[max_fact] = pow(fact[max_fact], MOD-2, MOD)
    for i in range(max_fact - 1, -1, -1):
        inv_fact[i] = inv_fact[i+1] * (i+1) % MOD

    max_exponent = 720
    overall = [0] * (max_exponent + 1)
    overall[0] = 1

    for d in '0123456789':
        d_int = int(d)
        c_d = freq[d]
        if c_d == 0:
            continue

        GF_d = [0] * (max_exponent + 1)
        for t in range(0, c_d + 1):
            if t > k:
                continue
            if d_int * t > S_even:
                continue

            if c_d < t:
                continue
            C = fact[c_d] * inv_fact[t] % MOD
            C = C * inv_fact[c_d - t] % MOD

            d_pow_t = pow(d_int, t, MOD)
            inv_t = inv_fact[t]

            term = C * d_pow_t % MOD
            term = term * inv_t % MOD
            if t > max_exponent:
                continue
            GF_d[t] = term

        new_overall = [0] * (max_exponent + 1)
        for i in range(len(overall)):
            if overall[i] == 0:
                continue
            for j in range(len(GF_d)):
                if GF_d[j] == 0:
                    continue
                if i + j > max_exponent:
                    continue
                new_overall[i + j] = (new_overall[i + j] + overall[i] * GF_d[j]) % MOD
        overall = new_overall

    coeff = overall[S_even] if S_even <= max_exponent else 0
    numerator = fact[k] * fact[n - k] % MOD
    result = (coeff * numerator) % MOD

    return result