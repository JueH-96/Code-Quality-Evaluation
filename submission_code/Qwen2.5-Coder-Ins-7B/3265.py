class Solution:
    def maximumSubarraySum(self, nums: List[int], k: int) -> int:
        max_sum = float('-inf')
        prefix_sum = {0: -1}
        current_sum = 0
        
        for i, num in enumerate(nums):
            current_sum += num
            
            if num + k in prefix_sum:
                max_sum = max(max_sum, current_sum - prefix_sum[num + k])
            if num - k in prefix_sum:
                max_sum = max(max_sum, current_sum - prefix_sum[num - k])
            
            if num not in prefix_sum:
                prefix_sum[num] = i
        
        return max_sum if max_sum != float('-inf') else 0