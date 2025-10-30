class Solution:
    def numberOfGoodPartitions(self, nums: List[int]) -> int:
        n = len(nums)
        mod = 10 ** 9 + 7
        
        # Create a dictionary to store the count of each number
        count = {}
        for num in nums:
            count[num] = count.get(num, 0) + 1
        
        # Create a dynamic programming table to store the number of good partitions
        dp = [0] * (n + 1)
        dp[0] = 1
        
        for i in range(1, n + 1):
            for num, freq in count.items():
                if freq <= i:
                    dp[i] = (dp[i] + dp[i - freq]) % mod
        
        return dp[n]