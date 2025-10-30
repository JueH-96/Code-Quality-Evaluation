class Solution:
    def minimumValueSum(self, nums: List[int], andValues: List[int]) -> int:
        n = len(nums)
        m = len(andValues)
        max_value = 10 ** 18  # a large number
        from collections import defaultdict

        dp = [ [defaultdict(lambda: max_value) for _ in range(m+2)] for _ in range(n)]
        # dp[pos][j][curr_and] = minimal total sum up to position pos, matched j andValues, ending with curr_and

        # Initialize dp[0]
        dp[0][0][nums[0]] = nums[0]
        if nums[0] == andValues[0]:
            dp[0][1][nums[0]] = nums[0]

        for pos in range(1, n):
            for j in range(m+1):
                # Extend previous subarrays
                for curr_and, total in dp[pos-1][j].items():
                    # Extend current subarray
                    new_and = curr_and & nums[pos]
                    new_sum = total + nums[pos]
                    if dp[pos][j][new_and] > new_sum:
                        dp[pos][j][new_and] = new_sum
                    # Check if new_and matches andValues[j], then we can end the subarray here
                    if j < m and new_and == andValues[j]:
                        if dp[pos][j+1][nums[pos]] > new_sum:
                            dp[pos][j+1][nums[pos]] = new_sum
                    # Start new subarray at pos
                    curr_and_start = nums[pos]
                    new_sum_start = total + nums[pos]
                    if dp[pos][j][curr_and_start] > new_sum_start:
                        dp[pos][j][curr_and_start] = new_sum_start
                    # Check if starting new subarray matches andValues[j]
                    if j < m and curr_and_start == andValues[j]:
                        if dp[pos][j+1][nums[pos]] > new_sum_start:
                            dp[pos][j+1][nums[pos]] = new_sum_start
            # Handle starting subarray from position pos (if we didn't extend any previous subarrays)
            # Only if pos == 0 or we start anew
            if pos == 0:
                if dp[pos][0][nums[pos]] > nums[pos]:
                    dp[pos][0][nums[pos]] = nums[pos]
                if nums[pos] == andValues[0]:
                    if dp[pos][1][nums[pos]] > nums[pos]:
                        dp[pos][1][nums[pos]] = nums[pos]

        # Find minimal total sum in dp[n-1][m]
        min_total = max_value
        for total in dp[n-1][m].values():
            if total < min_total:
                min_total = total

        if min_total == max_value:
            return -1
        else:
            return min_total