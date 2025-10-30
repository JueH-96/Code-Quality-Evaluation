class Solution:
    def countWays(self, nums):
        nums.sort()
        n = len(nums)
        mod = 10**9 + 7
        fact = [1]*(n+1)
        for i in range(1,n+1):
            fact[i] = (fact[i-1]*i)%mod
        def power(x,y):
            res = 1
            while y>0:
                if y&1:
                    res = (res*x)%mod
                y = y>>1
                x = (x*x)%mod
            return res
        def C(n,r):
            if r>n or r<0:
                return 0
            return (fact[n]*power(fact[r]*fact[n-r],mod-2))%mod
        ans = 0
        for i in range(n):
            ans = (ans+C(n,i+1))%mod
            if i<n-1 and nums[i+1]>i+1:
                break
        return ans