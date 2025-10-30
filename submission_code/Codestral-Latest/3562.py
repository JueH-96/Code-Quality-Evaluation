class Solution:
    def maximumWeight(self, intervals: List[List[int]]) -> List[int]:
        # Sort intervals by their end time
        intervals.sort(key=lambda x: x[1])

        # dp[i] will store the maximum weight we can get with i intervals
        dp = [[0, []] for _ in range(5)]
        for i, (start, end, weight) in enumerate(intervals):
            for j in range(4, 0, -1):
                # Find the latest non-overlapping interval
                k = bisect_right([intervals[idx][1] for idx in range(i)], start) - 1
                if k >= 0 and dp[k][0] + weight > dp[j][0]:
                    dp[j] = [dp[k][0] + weight, dp[k][1] + [i]]

        # Return the lexicographically smallest indices with maximum weight
        return dp[4][1]