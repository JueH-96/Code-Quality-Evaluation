from typing import List
class Solution:
    def subsequencesWithMiddleMode(self, nums: List[int]) -> int:
        MOD = 10**9 + 7
        n = len(nums)
        # fast small combinations
        def comb_n_k(n: int, k: int) -> int:
            if k < 0 or k > n: return 0
            if k == 0: return 1
            if k == 1: return n % MOD
            if k == 2: return (n * (n - 1) // 2) % MOD
            if k == 3: return (n * (n - 1) * (n - 2) // 6) % MOD
            if k == 4: return (n * (n - 1) * (n - 2) * (n - 3) // 24) % MOD
            # we only need up to k=4 here
            return 0
        
        # right counter init
        from collections import Counter
        freq_right = Counter(nums)
        freq_left = Counter()
        # sum of squares of counts
        sum_cl_sq = 0
        sum_cr_sq = sum(v*v for v in freq_right.values())
        left_count = 0
        right_count = n
        ans = 0
        
        for j, v in enumerate(nums):
            # move v from right to processing
            # decrement right_count
            right_count -= 1
            # update sum_cr_sq: remove old square, add new
            cr_old = freq_right[v]
            cr_new = cr_old - 1
            # update counter
            if cr_new > 0:
                freq_right[v] = cr_new
            else:
                del freq_right[v]
            # update sum of squares
            sum_cr_sq -= cr_old*cr_old
            sum_cr_sq += cr_new*cr_new
            
            # now freq_left holds counts on left, freq_right on right
            c_l_v = freq_left.get(v, 0)
            c_r_v = cr_new
            # totals on each side
            SL = left_count - c_l_v  # non-v on left
            SR = right_count - c_r_v  # non-v on right
            
            # sum of squares for non-v
            sum_cl_sq_nonv = sum_cl_sq - c_l_v*c_l_v
            sum_cr_sq_nonv = sum_cr_sq - c_r_v*c_r_v
            
            # C2 for non-v on each side
            C2_l = (SL*SL - sum_cl_sq_nonv) // 2
            C2_r = (SR*SR - sum_cr_sq_nonv) // 2
            
            # compute sum_IL_nonv = sum_{i!=v} c_r[i]*c_l[i]*(SL - c_l[i])
            sum_IL = 0
            # iterate over keys present in left; only those also in right matter
            for key, cl in freq_left.items():
                if key == v: continue
                cr = freq_right.get(key, 0)
                if cr:
                    sum_IL += cr * cl * (SL - cl)
            # compute sum_IR_nonv = sum_{i!=v} c_l[i]*c_r[i]*(SR - c_r[i])
            sum_IR = 0
            for key, cr in freq_right.items():
                if key == v: continue
                cl = freq_left.get(key, 0)
                if cl:
                    sum_IR += cl * cr * (SR - cr)
            
            # k = 2 case: need distinct 3 non-v picks
            # two scenarios: one extra v from left or from right
            # scenario A: pick one v from left, non-v TL=2,TR=1
            if c_l_v >= 1 and SL >= 2 and SR >= 1:
                # sum_L2R1 = SR * C2_l - sum_IL
                ways_nv = SR * C2_l - sum_IL
                ways_nv %= MOD
                ans = (ans + c_l_v * ways_nv) % MOD
            # scenario B: pick one v from right, non-v TL=1,TR=2
            if c_r_v >= 1 and SL >= 1 and SR >= 2:
                # sum_L1R2 = SL * C2_r - sum_IR
                ways_nv = SL * C2_r - sum_IR
                ways_nv %= MOD
                ans = (ans + c_r_v * ways_nv) % MOD
            
            # k = 3,4,5 cases
            # we always pick total 4 others: 2 from left, 2 from right
            # for k, pick (k-1) extra v's and (5-k) non-v's
            for k in (3,4,5):
                need_v = k - 1
                need_n = 5 - k
                # distribute v picks left/right
                # a_v_left + a_v_right = need_v
                # and a_v_left + a_n_left = 2, a_v_right + a_n_right = 2
                # so a_n_left = 2 - a_v_left, a_n_right = 2 - a_v_right
                for a_v_left in range(max(0, need_v - c_r_v), min(c_l_v, need_v) + 1):
                    a_v_right = need_v - a_v_left
                    # check counts
                    if a_v_right > c_r_v: continue
                    a_n_left = 2 - a_v_left
                    a_n_right = 2 - a_v_right
                    # check valid non-v picks
                    if a_n_left < 0 or a_n_left > SL: continue
                    if a_n_right < 0 or a_n_right > SR: continue
                    # ways for v
                    wv = comb_n_k(c_l_v, a_v_left) * comb_n_k(c_r_v, a_v_right) % MOD
                    # ways for non-v
                    wn = comb_n_k(SL, a_n_left) * comb_n_k(SR, a_n_right) % MOD
                    ans = (ans + wv * wn) % MOD
            
            # now move v to left side
            # update sum_cl_sq
            cl_old = c_l_v
            cl_new = cl_old + 1
            sum_cl_sq -= cl_old*cl_old
            sum_cl_sq += cl_new*cl_new
            freq_left[v] = cl_new
            left_count += 1
        
        return ans % MOD