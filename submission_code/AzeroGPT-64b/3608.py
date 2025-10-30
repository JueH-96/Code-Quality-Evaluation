class Solution:
    def subsequencePairCount(self, nums: List[int]) -> int:
        MOD = 10 ** 9 + 7
        from math import gcd
        from functools import lru_cache
        @lru_cache(None)
        def dp(i,g): # number of subsequences that can be formed using elemens up to i having gcd g
            if i>=len(nums): 
                if g>0:
                    return 1
                return 0
            return dp(i+1,g) + dp(i+1,gcd(g,nums[i]))
        moda = [0]*201
        for n in nums:
            for j,m in list(enumerate(moda))[-1:0:-1]:
                moda[gcd(m,n)]+=m
            moda[n] +=1
            
        ans=0
        for i,j in enumerate(moda):
            if not j: continue
            for k in range(201):
                if i==k: ans+=j*(j-1)//2
                elif gcd(i,k)==i: ans+=j*dp(0,k)
        return ans%MOD