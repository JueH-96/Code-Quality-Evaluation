class Solution:
    def maxSum(self, nums: List[int], k: int) -> int:
        n = len(nums)
        mod = 10**9 + 7
        max_val = max(nums) + 1
        freq = [0] * max_val
        dp = [0] * max_val
        for num in nums:
            freq[num] += 1
        
        dp[0] = 1  # Base case
        for i in range(1, max_val):
            dp[i] = (dp[i-1] * 2) % mod
            if freq[i]:
                dp[i] = (dp[i] + freq[i] - 1) % mod
        
        def calc_score(limit):
            max_score = 0
            for i in range(max_val):
                if freq[i]:
                    score = (dp[i] ** 2) * freq[i] % mod
                    if score <= limit:
                        max_score = max(max_score, score)
            return max_score
        
        # Find the k largest scores
        scores = sorted([calc_score(limit) for limit in range(k)], reverse=True)
        
        # Calculate the sum of squares
        result = sum([scores[i] * (i+1)**2 % mod for i in range(k)]) % mod
        return result