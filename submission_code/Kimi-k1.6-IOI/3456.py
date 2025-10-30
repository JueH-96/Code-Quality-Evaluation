class Solution:
    def maximumLength(self, nums: List[int], k: int) -> int:
        dp = [{} for _ in range(k + 1)]
        for num in nums:
            # Create a temporary DP as a copy of the current DP
            temp_dp = [d.copy() for d in dp]
            # Consider the case where the current num starts a new subsequence
            temp_dp[0][num] = max(temp_dp[0].get(num, 0), 1)
            # Update temp_dp based on previous states
            for c in range(k + 1):
                for x in dp[c]:
                    new_length = dp[c][x] + 1
                    if x == num:
                        # Transitions remain the same
                        if new_length > temp_dp[c].get(num, 0):
                            temp_dp[c][num] = new_length
                    else:
                        new_c = c + 1
                        if new_c <= k:
                            if new_length > temp_dp[new_c].get(num, 0):
                                temp_dp[new_c][num] = new_length
            # Update dp to be the new temp_dp
            dp = temp_dp
        # Find the maximum length across all states
        max_len = 0
        for c_dict in dp:
            if c_dict:
                current_max = max(c_dict.values())
                if current_max > max_len:
                    max_len = current_max
        return max_len