import sys
import threading
def main():
    import sys
    sys.setrecursionlimit(1 << 25)
    MOD = 998244353

    S = sys.stdin.readline().strip()
    if not S:
        return
    N = len(S)

    # Precompute prefix arrays Q_arr, L_arr, U_arr
    Q_arr = [0] * (N + 1)  # count of '?' in prefix
    L_arr = [0] * (N + 1)  # count of fixed lowercase in prefix
    U_arr = [0] * (N + 1)  # count of fixed uppercase in prefix
    for i, ch in enumerate(S, start=1):
        q = Q_arr[i-1] + (1 if ch == '?' else 0)
        Q_arr[i] = q
        if 'a' <= ch <= 'z':
            L_arr[i] = L_arr[i-1] + 1
        else:
            L_arr[i] = L_arr[i-1]
        if 'A' <= ch <= 'Z':
            U_arr[i] = U_arr[i-1] + 1
        else:
            U_arr[i] = U_arr[i-1]

    # Precompute factorials and invfacts up to max(N, 26)
    maxF = N if N > 26 else 26
    fact = [1] * (maxF + 1)
    for i in range(1, maxF + 1):
        fact[i] = fact[i-1] * i % MOD
    invfact = [1] * (maxF + 1)
    invfact[maxF] = pow(fact[maxF], MOD-2, MOD)
    for i in range(maxF, 0, -1):
        invfact[i-1] = invfact[i] * i % MOD

    # Precompute powers of 26 and inv26
    pow26 = [1] * (N + 1)
    for i in range(1, N + 1):
        pow26[i] = pow26[i-1] * 26 % MOD
    inv26 = pow(26, MOD-2, MOD)
    inv26Pow = [1] * (N + 1)
    for i in range(1, N + 1):
        inv26Pow[i] = inv26Pow[i-1] * inv26 % MOD

    # Prepare DP
    # sums[k] = sum of D_lower[T] * inv26^Q_arr[T] for T processed where L_arr[T] == k
    sums = [0] * (N + 1)
    # Initialize for T=0
    # We treat D_lower[0] = 1, Q_arr[0]=0, L_arr[0]=0
    sums[0] = 1  # D_lower[0] * inv26Pow[0] = 1

    # Track fixed uppercase duplicates and count distinct fixed uppercase
    seen_upper = [False] * 26
    fixed_dup = False
    distinct_fixed_up = 0

    total_safe = 0
    D_prev = 1  # D[0]

    # Pre-calc for P(a,j): fact[a] * invfact[a-j]
    # But a <= 26, j <= a
    # We'll compute on the fly: P(a,j) = fact[a] * invfact[a-j] % MOD

    # Pre-calc binomial via fact and invfact

    # Loop over positions
    last_fixed_up_count = U_arr[N]  # total fixed uppercase in entire S
    for i in range(1, N + 1):
        ch = S[i-1]
        # D_lower[i] : assignments to prefix 1..i with no duplicate uppercase and s[i] lowercase
        if ch == '?':
            # '?' assigned lowercase: 26 choices
            Dl = D_prev * 26 % MOD
        elif 'a' <= ch <= 'z':
            # fixed lowercase: 1 choice
            Dl = D_prev
        else:
            # fixed uppercase: cannot be lowercase
            Dl = 0

        # Compute W[i]: safe assignments for prefix ending at i with s[i] uppercase and no bad subseq
        # W[i] = pow26[Q_arr[i]] * sums[ L_arr[i] ] (before we add Dl for T=i)
        if 'a' <= ch <= 'z':
            Wi = 0
        else:
            # pow26 for Q_arr[i]
            Wi = pow26[Q_arr[i]] * sums[L_arr[i]] % MOD

        # Suffix multiplier: suffix positions i+1..N all lowercase
        # Only if no fixed uppercase in suffix
        if last_fixed_up_count - U_arr[i] == 0:
            # all '?' in suffix must be lowercase: 26 choices each
            suf_q = Q_arr[N] - Q_arr[i]
            total_safe = (total_safe + Wi * pow26[suf_q]) % MOD
        # else multiplier is zero, skip

        # Now update fixed uppercase prefix state before computing D[i]
        if 'A' <= ch <= 'Z':
            idx = ord(ch) - ord('A')
            if not fixed_dup:
                if seen_upper[idx]:
                    fixed_dup = True
                else:
                    seen_upper[idx] = True
                    distinct_fixed_up += 1

        # Compute D[i]: assignments to prefix 1..i with no duplicate uppercase (no constraint on s[i])
        if fixed_dup:
            Di = 0
        else:
            # compute D[i] = sum_{j=0..min(Q_i, M)} C(Q_i, j)*P(M,j)*26^{Q_i - j}
            Qi = Q_arr[i]
            M = 26 - distinct_fixed_up
            # j up to min(Qi, M)
            # M could be negative if distinct_fixed_up >26, but that only if fixed_dup, but fixed_dup true => we skip here
            maxj = M if M < Qi else Qi
            # sum loop
            s_val = 0
            # locals for speed
            fact_loc = fact
            invfact_loc = invfact
            pow26_loc = pow26
            for j in range(maxj + 1):
                # binomial Qi choose j
                # C = fact[Qi] * invfact[j] * invfact[Qi-j]
                c = fact_loc[Qi] * (invfact_loc[j] * invfact_loc[Qi - j] % MOD) % MOD
                # P(M, j) = fact[M] * invfact[M-j]
                p = fact_loc[M] * invfact_loc[M - j] % MOD
                # term = C * p * 26^{Qi - j}
                s_val = (s_val + c * p % MOD * pow26_loc[Qi - j]) % MOD
            Di = s_val
        # Prepare for next iteration
        D_prev = Di

        # Add D_lower[i] to sums group L_arr[i]
        # val = D_lower[i] * inv26Pow[ Q_arr[i] ]
        val = Dl * inv26Pow[Q_arr[i]] % MOD
        sums[L_arr[i]] = (sums[L_arr[i]] + val) % MOD

    # Add L=0 case: no uppercase anywhere => all '?' lowercase
    if last_fixed_up_count == 0:
        # one assignment to prefix empty, multiply suffix of whole string q '?' => 26^q
        total_safe = (total_safe + pow26[Q_arr[N]]) % MOD

    print(total_safe)

if __name__ == "__main__":
    main()