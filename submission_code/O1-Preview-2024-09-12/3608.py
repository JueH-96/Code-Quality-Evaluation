class Solution:
    def subsequencePairCount(self, nums: List[int]) -> int:
        from math import gcd
        MOD = 10**9+7
        
        max_num = max(nums)
        count = [0]*(max_num+1)
        for num in nums:
            count[num] +=1

        f = [0]*(max_num+1)
        for i in range(max_num, 0, -1):
            total = 0
            cnt = 0
            for j in range(i, max_num+1, i):
                cnt += count[j]
            if cnt ==0:
                continue
            total_subseq = pow(2, cnt, MOD) -1
            for k in range(2*i, max_num+1, i):
                total_subseq = (total_subseq - f[k]) % MOD
            f[i] = total_subseq

        ans = 0
        for g in range(1, max_num+1):
            s = f[g]
            ans = (ans + s * s) % MOD

        return ans