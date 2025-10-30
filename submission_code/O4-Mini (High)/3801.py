class Solution:
    def beautifulNumbers(self, l: int, r: int) -> int:
        import math

        # Precompute lengths and powers up to length of r
        sr = str(r)
        max_n = len(sr)
        # pow9[i] = 9^i, pow10[i] = 10^i
        pow9 = [1] * (max_n + 1)
        pow10 = [1] * (max_n + 1)
        for i in range(1, max_n + 1):
            pow9[i] = pow9[i - 1] * 9
            pow10[i] = pow10[i - 1] * 10

        # Build list of allowed sums: 7-smooth numbers <= 9*max_n
        max_sum = 9 * max_n
        allowed = set()
        # exponents for 2,3,5,7
        for a in range(0, 7):  # 2^6 = 64 <= 81
            v2 = 2**a
            if v2 > max_sum:
                break
            for b in range(0, 5):  # 3^4 = 81
                v23 = v2 * (3**b)
                if v23 > max_sum:
                    break
                for c in range(0, 3):  # 5^2 = 25, 5^3=125>81
                    v235 = v23 * (5**c)
                    if v235 > max_sum:
                        break
                    for d in range(0, 3):  # 7^2 = 49, 7^3=343>81
                        v = v235 * (7**d)
                        if v > max_sum:
                            break
                        allowed.add(v)
        allowed_sums = sorted(allowed)

        # Precompute suffix DP: dp_suf_mod[S][L][s][r] = count of length-L sequences,
        # digits 1..9, sum = s, product mod S = r.
        # We only need L up to max_n-1.
        maxL = max_n - 1
        dp_suf_mod = {}
        for S in allowed_sums:
            # dp_list[L] will be a 2D array of shape (9*L+1) x S
            dp_list = []
            # L = 0: only the empty sequence, sum=0, product=1 => rem = 1 % S
            dp0 = [[0] * S for _ in range(1)]
            dp0[0][1 % S] = 1
            dp_list.append(dp0)
            # Build for L = 1..maxL
            for L in range(1, maxL + 1):
                prev = dp_list[L - 1]
                smax = 9 * L
                dpL = [[0] * S for _ in range(smax + 1)]
                # Transition from length L-1 to L by appending d in 1..9
                for s_prev in range(len(prev)):
                    row = prev[s_prev]
                    # iterate remainders
                    for r_prev, cnt in enumerate(row):
                        if not cnt:
                            continue
                        # append digit d
                        for d in range(1, 10):
                            dpL[s_prev + d][(r_prev * d) % S] += cnt
                dp_list.append(dpL)
            dp_suf_mod[S] = dp_list

        def count_beautiful_upto(X: int) -> int:
            # Count beautiful numbers in [1..X]
            if X <= 0:
                return 0
            ds = list(map(int, str(X)))
            n = len(ds)
            total = 0

            # 1) full lengths < n
            for k in range(1, n):
                # count numbers of length k with at least one zero
                # total k-digit numbers = 9 * 10^(k-1)
                # no-zero k-digit numbers = 9^k
                total += 9 * pow10[k - 1] - pow9[k]
                # count no-zero k-digit numbers with product divisible by sum
                for S in allowed_sums:
                    if S > 9 * k:
                        break
                    # number of sequences of length k, sum=S, rem=0
                    total += dp_suf_mod[S][k][S][0]

            # 2) length = n
            # 2a) count no-zero numbers <= X (classic digit-DP for zeros)
            cnt_no_zero = 0
            for i, d in enumerate(ds):
                if d == 0:
                    break
                # choose a smaller digit here (1..d-1), then any of 1..9 in suffix
                cnt_no_zero += (d - 1) * pow9[n - i - 1]
            else:
                # if we never hit a zero digit, X itself is no-zero
                cnt_no_zero += 1
            # total n-digit numbers = X - 10^(n-1) + 1
            total += (X - pow10[n - 1] + 1) - cnt_no_zero

            # 2b) among no-zero n-digit numbers <= X, count those with product%sum==0
            for S in allowed_sums:
                if S > 9 * n:
                    break
                # prefix DP, only one prefix state since must match X until break
                prefix_sum = 0
                prefix_rem = 1 % S
                ok = True
                resS = 0
                for pos in range(n):
                    bound = ds[pos]
                    # digits < bound => branch into suffix DP
                    maxd = bound - 1
                    if maxd >= 1:
                        L = n - pos - 1
                        dp_list = dp_suf_mod[S]
                        for d in range(1, maxd + 1):
                            ns = prefix_sum + d
                            if ns > S:
                                break
                            r2 = (prefix_rem * d) % S
                            s2 = S - ns
                            if s2 < 0 or s2 > 9 * L:
                                continue
                            # need product_suf % S to satisfy (r2 * suf) % S == 0
                            g = math.gcd(r2, S)
                            m = S // g
                            row = dp_list[L][s2]
                            # sum over multiples of m
                            for rem in range(0, S, m):
                                resS += row[rem]
                    # now choose d == bound to continue prefix
                    if bound == 0:
                        ok = False
                        break
                    prefix_sum += bound
                    if prefix_sum > S:
                        ok = False
                        break
                    prefix_rem = (prefix_rem * bound) % S
                # if prefix matched all digits and satisfies at end
                if ok and prefix_sum == S and prefix_rem == 0:
                    resS += 1
                total += resS

            return total

        return count_beautiful_upto(r) - count_beautiful_upto(l - 1)