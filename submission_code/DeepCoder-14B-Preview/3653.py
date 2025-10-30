class Solution:
    def maxSubarraySum(self, nums: List[int], k: int) -> int:
        prefix = [0]
        for num in nums:
            prefix.append(prefix[-1] + num)
        
        min_prefix = [float('inf')] * k
        min_prefix[0] = prefix[0]  # Initialize with the first prefix sum
        
        max_sum = -float('inf')
        
        for m in range(1, len(prefix)):
            r = m % k
            if min_prefix[r] != float('inf'):
                current_sum = prefix[m] - min_prefix[r]
                if current_sum > max_sum:
                    max_sum = current_sum
            # Update the minimum prefix for the current remainder
            if prefix[m] < min_prefix[r]:
                min_prefix[r] = prefix[m]
        
        return max_sum if max_sum != -float('inf') else 0