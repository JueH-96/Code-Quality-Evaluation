class Solution:
    def findMaximumLength(self, nums: List[int]) -> int:
        n = len(nums)
        # Compute prefix sums for efficient range sum queries
        prefix = [0]
        for num in nums:
            prefix.append(prefix[-1] + num)
        
        # dp[i] = (max_length, min_last_value) for elements nums[0:i]
        dp = [(0, 0)]  # Base case: empty array has length 0 and last value 0
        
        for i in range(1, n + 1):
            max_len = 0
            min_last = float('inf')
            
            # Try all possible starting positions for the last segment
            for j in range(i):
                segment_sum = prefix[i] - prefix[j]
                prev_len, prev_last = dp[j]
                
                # Check if we can extend the sequence
                if segment_sum >= prev_last:
                    new_len = prev_len + 1
                    if new_len > max_len:
                        max_len = new_len
                        min_last = segment_sum
                    elif new_len == max_len:
                        min_last = min(min_last, segment_sum)
            
            dp.append((max_len, min_last))
        
        return dp[n][0]