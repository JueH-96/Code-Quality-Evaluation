class Solution:
    def maximumSubarraySumGood(self, nums: List[int], k: int) -> int:
        max_sum = -float('inf')
        min_prefix = {}
        prefix_sum = [0]
        for num in nums:
            prefix_sum.append(prefix_sum[-1] + num)
        
        for j in range(len(nums)):
            current = nums[j]
            target1 = current - k
            target2 = current + k
            
            possible_sum = prefix_sum[j] - min(
                min_prefix.get(target1, float('inf')),
                min_prefix.get(target2, float('inf'))
            )
            
            if possible_sum > max_sum:
                max_sum = possible_sum
            
            # Update min_prefix for current
            if current in min_prefix:
                if prefix_sum[j] < min_prefix[current]:
                    min_prefix[current] = prefix_sum[j]
            else:
                min_prefix[current] = prefix_sum[j]
        
        return max_sum if max_sum != -float('inf') else 0