class Solution:
    def numberOfSubsequences(self, nums: List[int]) -> int:
        from collections import defaultdict

        # Dictionary to store the count of each number
        count = defaultdict(int)
        for num in nums:
            count[num] += 1

        # Sort the unique numbers
        unique_nums = sorted(count.keys())

        # Dictionary to store the number of valid subsequences ending with each unique number
        dp = defaultdict(int)

        # Iterate over each unique number
        for num in unique_nums:
            # If the number appears at least 4 times, it can form a subsequence by itself
            if count[num] >= 4:
                dp[num] += 1

            # Iterate over each previous unique number
            for prev_num in unique_nums:
                if prev_num == num:
                    break
                if count[prev_num] >= 2:
                    dp[num] += dp[prev_num]

        # Sum up all the valid subsequences
        result = sum(dp.values())

        return result