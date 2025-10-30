class Solution:
    def minimumCost(self, s: str) -> int:
        # We derive a closed‐form solution via XOR‐linearization.
        # For each possible final bit T (0 or 1), we compute the minimum cost
        # to flip s into all T's.  Return the smaller of the two.
        n = len(s)
        # Pre‐convert s into ints 0/1
        sval = [1 if ch == '1' else 0 for ch in s]
        INF = 10**30

        def solve_for_target(T: int) -> int:
            # d[j] = s[j] xor T
            d0 = sval[0] ^ T
            # build d
            d = [ (sv ^ T) for sv in sval ]
            # build delta array of length n-1: delta[i] = d[i] xor d[i+1], for i=0..n-2
            # records where d changes
            delta = [ (d[i] ^ d[i+1]) for i in range(n-1) ]

            # C0 = sum_{i=1..n-1} (n-i) * delta[i-1]
            # since delta[i-1] corresponds to Δ_i in the writeup
            C0 = 0
            # i runs 1..n-1, delta index i-1
            # weight = n - i
            for i in range(1, n):
                if delta[i-1]:
                    C0 += (n - i)

            # Build the linear coefficients c[i]
            # For i=0..n-2: c[i] = (i+1) + (n-(i+1))*(1 - 2*delta[i])
            # For i=n-1: c[n-1] = n
            sum_neg = 0
            count_neg = 0
            min_pos = INF
            min_rem = INF

            # i = 0..n-2
            for i in range(n-1):
                # delta[i] = d[i] xor d[i+1]
                di = delta[i]
                # c_i:
                #   termA = i+1
                #   termB = (n-(i+1)) * (1 - 2*di)
                c = (i+1) + (n-(i+1)) * (1 - 2*di)
                if c < 0:
                    sum_neg += c
                    count_neg += 1
                    # removal cost = -c
                    if -c < min_rem:
                        min_rem = -c
                else:
                    # addition cost = c
                    if c < min_pos:
                        min_pos = c

            # i = n-1
            # only termA = n
            c_last = n
            # c_last >= 0 always
            if c_last < min_pos:
                min_pos = c_last

            # now we have:
            # sum_neg = sum of all negative c_i
            # count_neg = number of negative c_i
            # min_pos = minimal c_i >= 0
            # min_rem = minimal removal cost for negative c_i
            par_neg = count_neg & 1  # parity of number of chosen p_i=1
            best_raw = INF
            # try both parity goals p_sum mod2 = 0 or 1
            for want_par in (0, 1):
                # cost to adjust parity if par_neg != want_par
                if par_neg == want_par:
                    adj = 0
                else:
                    # we can either add one positive-c (cost min_pos)
                    # or remove one negative-c (cost min_rem)
                    adj = min(min_pos, min_rem)
                # penalty if the chosen parity want_par doesn't match d0:
                # penalty = n if want_par != d0 else 0
                pen = n if (want_par != d0) else 0
                cost_raw = sum_neg + adj + pen
                if cost_raw < best_raw:
                    best_raw = cost_raw

            # total cost for this T = C0 + best_raw + const_term
            # const_term is baked into C0 and the derivation; no further add
            return C0 + best_raw

        # solve for T=0 and T=1
        cost0 = solve_for_target(0)
        cost1 = solve_for_target(1)
        return min(cost0, cost1)