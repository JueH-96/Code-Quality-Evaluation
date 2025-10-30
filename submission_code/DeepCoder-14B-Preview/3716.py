class Solution:
    def longestSubsequence(self, nums: List[int]) -> int:
        dp = {}  # key: last difference, value: max length
        max_len = 1  # at least, each element is a subsequence of length 1
        
        for i in range(len(nums)):
            current_num = nums[i]
            # Create a copy of the current dp to use for this i
            current_dp = dp.copy()
            temp_dp = {}
            
            # Consider all previous elements
            for j in range(i):
                prev_num = nums[j]
                d = abs(current_num - prev_num)
                
                # Option 1: start a new subsequence with j and i
                if d in temp_dp:
                    if 2 > temp_dp[d]:
                        temp_dp[d] = 2
                else:
                    temp_dp[d] = 2
                
                # Option 2: extend existing subsequences
                for d_prev in current_dp:
                    if d <= d_prev:
                        new_length = current_dp[d_prev] + 1
                        if d in temp_dp:
                            if new_length > temp_dp[d]:
                                temp_dp[d] = new_length
                        else:
                            temp_dp[d] = new_length
            
            # Merge temp_dp into dp
            for d in temp_dp:
                if d in dp:
                    if temp_dp[d] > dp[d]:
                        dp[d] = temp_dp[d]
                else:
                    dp[d] = temp_dp[d]
            
            # Update the max_len
            current_max = max(dp.values(), default=0)
            if current_max > max_len:
                max_len = current_max
        
        return max_len