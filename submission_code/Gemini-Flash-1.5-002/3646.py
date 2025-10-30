class Solution:
    def sumOfGoodSubsequences(self, nums: List[int]) -> int:
        mod = 10**9 + 7
        count = {}
        for num in nums:
            count[num] = count.get(num, 0) + 1
        
        dp = {}
        def solve(num):
            if num in dp:
                return dp[num]
            
            total_sum = count.get(num,0) * num
            
            if num + 1 in count:
                total_sum = (total_sum + (count.get(num,0) * solve(num+1))) % mod
            if num -1 in count and num > 0:
                total_sum = (total_sum + (count.get(num,0) * solve(num-1))) % mod

            dp[num] = total_sum
            return total_sum

        total_sum = 0
        for num in count:
            total_sum = (total_sum + solve(num)) % mod

        return total_sum