class Solution:
    def countOfPairs(self, nums: List[int]) -> int:
        n = len(nums)
        MOD = 10**9 + 7
        fact = [1]*(n+1)
        inv = [1]*(n+1)
        for i in range(1, n+1):
            fact[i] = (fact[i-1]*i)%MOD
            inv[i] = pow(fact[i], MOD-2, MOD)
        comb = lambda x, y: 0 if y > x else (fact[x]*inv[y]*inv[x-y])%MOD
        nums.sort()
        ans = 0
        i, j = 0, n-1
        while i <= j:
            if nums[i] + nums[j] <= 5:
                if nums[i] == nums[j]:
                    ans = (ans + comb(j-i+1, 2))%MOD
                    i += 1
                    j -= 1
                else:
                    k = 1
                    while i+k < n and nums[i+k] == nums[i]:
                        k += 1
                    l = 1
                    while j-l >= 0 and nums[j-l] == nums[j]:
                        l += 1
                    ans = (ans + comb(k, 1)*comb(l, 1))%MOD
                    i += k
                    j -= l
            else:
                j -= 1
        return ans