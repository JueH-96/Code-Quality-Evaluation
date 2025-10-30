class Solution:
    def lengthOfLongestSubsequence(self, nums: List[int], target: int) -> int:
        # Initialize a dictionary to store the minimum length of subsequence to achieve each sum
        dp = {0: 0}

        max_length = -1

        for num in nums:
            # Create a new dictionary to avoid overwriting values during iteration
            new_dp = dp.copy()
            for current_sum, length in dp.items():
                new_sum = current_sum + num
                if new_sum == target:
                    max_length = max(max_length, length + 1)
                elif new_sum < target:
                    if new_sum not in new_dp or new_dp[new_sum] > length + 1:
                        new_dp[new_sum] = length + 1
            dp = new_dp

        return max_length