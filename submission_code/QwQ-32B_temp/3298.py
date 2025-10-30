class Solution:
    def maxSelectedElements(self, nums: List[int]) -> int:
        nums.sort()
        dp = {}
        max_len = 0
        for num in nums:
            a = num
            b = num + 1
            
            # Calculate new possible lengths for a and b
            new_a = dp.get(a - 1, 0) + 1
            new_b = dp.get(b - 1, 0) + 1
            
            # Update the dp dictionary for a and b if the new values are better
            current_a = dp.get(a, 0)
            if new_a > current_a:
                dp[a] = new_a
            else:
                # Remove a from dp if it's worse than existing, but not necessary here
                pass  # We can leave it as is since we only track maximums
            
            current_b = dp.get(b, 0)
            if new_b > current_b:
                dp[b] = new_b
            else:
                pass  # Similarly, leave as is
            
            # Update the maximum length found so far
            current_max = max(new_a, new_b)
            if current_max > max_len:
                max_len = current_max
        
        return max_len