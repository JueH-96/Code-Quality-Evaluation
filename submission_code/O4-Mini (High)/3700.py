class Solution:
    def subsequencesWithMiddleMode(self, nums: List[int]) -> int:
        mod = 10**9 + 7
        n = len(nums)
        # compress values to 0..m-1
        vals = list(set(nums))
        vals.sort()
        comp = {v:i for i,v in enumerate(vals)}
        arr = [comp[v] for v in nums]
        m = len(vals)
        # total counts on right initially
        total = [0] * m
        for v in arr:
            total[v] += 1
        # d_left and d_right
        d_left = [0] * m
        d_right = total[:]  # counts in suffix including current k initially
        # initialize sums
        sum_AB_all = 0             # sum d_left[u] * d_right[u]
        sum_AB2_all = 0            # sum d_left[u] * (d_right[u]^2)
        sum_C2_right_all = 0       # sum C(d_right[u],2)
        sum_C2_left_all = 0        # sum C(d_left[u],2)
        sum_A2B_all = 0            # sum (d_left[u]^2 * d_right[u])
        # compute initial sum_C2_right_all
        for u in range(m):
            b = d_right[u]
            sum_C2_right_all = (sum_C2_right_all + b * (b - 1) // 2) % mod
        ans = 0
        # iterate possible middle index k
        for k in range(n):
            v = arr[k]
            # remove position k from right side
            old_rv = d_right[v]
            # update sums to reflect d_right[v] -> old_rv-1
            # sum_AB_all: change = d_left[v] * (new_rv - old_rv) = -d_left[v]
            sum_AB_all = (sum_AB_all - d_left[v]) % mod
            # sum_AB2_all: change = d_left[v] * ((new_rv)^2 - old_rv^2)
            # = d_left[v] * ((old_rv-1)^2 - old_rv^2) = -(2*old_rv -1)*d_left[v]
            delta = (2 * old_rv - 1) % mod
            sum_AB2_all = (sum_AB2_all - d_left[v] * delta) % mod
            # sum_C2_right_all: change = C(old_rv-1,2) - C(old_rv,2) = -old_rv + 1
            sum_C2_right_all = (sum_C2_right_all - (old_rv - 1)) % mod
            # sum_A2B_all: change = d_left[v]^2 * (new_rv - old_rv) = -d_left[v]^2
            sum_A2B_all = (sum_A2B_all - d_left[v] * d_left[v]) % mod
            # decrement
            d_right[v] = old_rv - 1
            # compute counts
            L1 = d_left[v]
            R1 = d_right[v]
            L0 = k - L1
            R0 = (n - 1 - k) - R1
            # precompute small combs
            C2_L0 = L0 * (L0 - 1) // 2
            C2_L1 = L1 * (L1 - 1) // 2
            C2_R0 = R0 * (R0 - 1) // 2
            C2_R1 = R1 * (R1 - 1) // 2
            # ways_left[a], ways_right[b]
            wl0 = C2_L0
            wl1 = L1 * L0
            wl2 = C2_L1
            wr0 = C2_R0
            wr1 = R1 * R0
            wr2 = C2_R1
            # pure cases a+b>=2
            sum_pure = (
                wl2 * wr0 + wl0 * wr2 + wl1 * wr1
                + wl2 * wr1 + wl1 * wr2 + wl2 * wr2
            ) % mod
            # case (a,b) = (1,0): one v on left, one other on left, two others on right
            # D1 = # triples of one left-other and two right-others with distinct values
            # total triples = L0 * C2(R0)
            total_triples1 = L0 * C2_R0 % mod
            # sums excluding u=v
            # sum_AB = sum_{u!=v} d_left[u]*d_right[u]
            sum_AB = (sum_AB_all - d_left[v] * d_right[v]) % mod
            # sum_A_B2 = sum_{u!=v} d_left[u] * (d_right[u]^2)
            sum_A_B2 = (sum_AB2_all - d_left[v] * (d_right[v] * d_right[v] % mod)) % mod
            # sum_C_B2 = sum_{u!=v} C(d_right[u],2)
            sum_C_B2 = (sum_C2_right_all - (d_right[v] * (d_right[v] - 1) // 2)) % mod
            # bad triples
            total_bad1 = (
                (R0 % mod) * sum_AB % mod
                - sum_A_B2
                + (L0 % mod) * sum_C_B2 % mod
            ) % mod
            D1 = (total_triples1 - total_bad1) % mod
            good1 = L1 * D1 % mod
            # case (a,b) = (0,1): two others on left, one other on right
            total_triples2 = C2_L0 * R0 % mod
            # sum_C_A2 = sum_{u!=v} C(d_left[u],2)
            sum_C_A2 = (sum_C2_left_all - (d_left[v] * (d_left[v] - 1) // 2)) % mod
            # sum_A2_B = sum_{u!=v} (d_left[u]^2 * d_right[u])
            sum_A2_B = (sum_A2B_all - (d_left[v] * d_left[v] % mod) * d_right[v]) % mod
            total_bad2 = (
                (R0 % mod) * sum_C_A2 % mod
                + (L0 % mod) * sum_AB % mod
                - sum_A2_B
            ) % mod
            D2 = (total_triples2 - total_bad2) % mod
            good2 = R1 * D2 % mod
            # accumulate
            ans = (ans + sum_pure + good1 + good2) % mod
            # now add position k to left side: d_left[v] += 1
            old_lv = d_left[v]
            # update sums
            # sum_AB_all: add d_right[v]
            sum_AB_all = (sum_AB_all + d_right[v]) % mod
            # sum_AB2_all: add d_right[v]^2
            sum_AB2_all = (sum_AB2_all + d_right[v] * d_right[v] % mod) % mod
            # sum_C2_left_all: add C(old_lv+1,2)-C(old_lv,2) = old_lv
            sum_C2_left_all = (sum_C2_left_all + old_lv) % mod
            # sum_A2B_all: add ((old_lv+1)^2 - old_lv^2)*d_right[v] = (2*old_lv+1)*d_right[v]
            sum_A2B_all = (sum_A2B_all + (2 * old_lv + 1) * d_right[v]) % mod
            # increment
            d_left[v] = old_lv + 1
        return ans