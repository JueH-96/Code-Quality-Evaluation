class Solution:
    def sumOfGoodSubsequences(self, nums: List[int]) -> int:
        MOD = 10**9 + 7
        if not nums:
            return 0
        
        # Maximum possible value in nums is up to 10^5
        MAXV = 10**5 + 2
        cnt = [0] * MAXV  # cnt[x]: number of good subseq ending with x
        sm = [0] * MAXV   # sm[x]: total sum of elements over those subseqs
        
        for x in nums:
            # number/sum of subseqs ending with x-1 and x+1
            c1, s1 = cnt[x-1], sm[x-1] if x-1 >= 0 else (0, 0)
            c2, s2 = cnt[x+1], sm[x+1] if x+1 < MAXV else (0, 0)
            
            # ways to extend from neighbors
            extend_cnt = (c1 + c2) % MOD
            extend_sum = (s1 + c1 * x + s2 + c2 * x) % MOD
            
            # singleton subsequence [x]
            new_cnt = (extend_cnt + 1) % MOD
            new_sum = (extend_sum + x) % MOD
            
            # accumulate into dp for x
            cnt[x] = (cnt[x] + new_cnt) % MOD
            sm[x] = (sm[x] + new_sum) % MOD
        
        # total sum over all good subsequences
        return sum(sm) % MOD