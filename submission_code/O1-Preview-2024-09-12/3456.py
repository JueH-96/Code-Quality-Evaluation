class Solution:
    def maximumLength(self, nums: List[int], k: int) -> int:
        n = len(nums)
        set_values = list(set(nums))
        value_to_index = {value: index for index, value in enumerate(set_values)}
        nums_indices = [value_to_index[num] for num in nums]
        num_unique = len(set_values)
        dp = [{} for _ in range(k+2)]
        dp[0][None] = 0  # Initialize with no last_value
        for idx, num_idx in enumerate(nums_indices):
            num = nums[idx]
            new_dp = [{} for _ in range(k+2)]
            for changes in range(k+1):
                for last_value in dp[changes]:
                    curr_len = dp[changes][last_value]
                    # Option 1: Do not include nums[idx]
                    if last_value in new_dp[changes]:
                        new_dp[changes][last_value] = max(new_dp[changes][last_value], curr_len)
                    else:
                        new_dp[changes][last_value] = curr_len
                    # Option 2: Include nums[idx]
                    if last_value is None or set_values[last_value] == num:
                        # No change in value
                        if num_idx in new_dp[changes]:
                            new_dp[changes][num_idx] = max(new_dp[changes][num_idx], curr_len +1)
                        else:
                            new_dp[changes][num_idx] = curr_len +1
                    else:
                        # Change in value
                        if changes+1 <= k:
                            if num_idx in new_dp[changes+1]:
                                new_dp[changes+1][num_idx] = max(new_dp[changes+1][num_idx], curr_len +1)
                            else:
                                new_dp[changes+1][num_idx] = curr_len +1
            dp = new_dp
        max_length = 0
        for changes in range(k+1):
            for length in dp[changes].values():
                if length > max_length:
                    max_length = length
        return max_length