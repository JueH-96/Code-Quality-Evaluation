class Solution:
    def countWays(self, nums: List[int]) -> int:
        n = len(nums)
        if n == 0:
            return 0
        
        max_s = max(nums) if nums else 0
        freq = [0] * (n + 1)
        for num in nums:
            if num <= n:
                freq[num] += 1
        
        # Compute prefix sum array
        prefix = [0] * (n + 1)
        for s in range(1, n + 1):
            prefix[s] = prefix[s - 1] + freq[s]
        
        ans = 0
        for s in range(n + 1):
            m = prefix[s]
            if m >= s and freq[s] == 0:
                ans += 1
        
        return ans