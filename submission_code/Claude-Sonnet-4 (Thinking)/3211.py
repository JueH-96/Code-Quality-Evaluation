class Solution:
    def findMaximumLength(self, nums: List[int]) -> int:
        n = len(nums)
        prefix = [0]
        for num in nums:
            prefix.append(prefix[-1] + num)
        
        # dp[i] = dictionary mapping num_segments -> min_last_sum
        dp = [{} for _ in range(n + 1)]
        dp[0][0] = 0
        
        for i in range(1, n + 1):
            for j in range(i):  # last segment is [j, i-1]
                segment_sum = prefix[i] - prefix[j]
                for num_segments, min_last_sum in dp[j].items():
                    if segment_sum >= min_last_sum:
                        new_segments = num_segments + 1
                        if new_segments not in dp[i] or dp[i][new_segments] > segment_sum:
                            dp[i][new_segments] = segment_sum
        
        return max(dp[n].keys())