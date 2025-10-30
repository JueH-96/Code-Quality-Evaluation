class Solution:
    def sumOfPowers(self, nums: List[int], k: int) -> int:
        n = len(nums)
        mod = 10**9 + 7
        nums.sort()
        
        def nCr(n, r):
            if r < 0 or r > n:
                return 0
            if r == 0 or r == n:
                return 1
            if r > n // 2:
                r = n - r
            
            result = 1
            for i in range(r):
                result = result * (n - i) // (i + 1)
            return result

        ans = 0
        for i in range(n):
            for j in range(i + 1, n):
                diff = abs(nums[j] - nums[i])
                count = nCr(i, k - 2) + nCr(j - i - 1, k - 2)
                
                
                ans = (ans + diff * (nCr(n - j -1 , k-2) + nCr(j-i-1, k-2) + nCr(i,k-2))) % mod
                
        
        if k == 1:
            ans = sum(nums)%mod
            
        elif k==2:
            ans = 0
            for i in range(n):
                for j in range(i+1, n):
                    ans = (ans + abs(nums[i]-nums[j]))%mod
        
        return ans