class Solution:
    def getLargestOutlier(self, nums: List[int]) -> int:
        total_sum = sum(nums)
        max_outlier = float('-inf')
        
        # Create a frequency map
        freq = {}
        for num in nums:
            freq[num] = freq.get(num, 0) + 1
        
        for num in nums:
            # Consider num as potential outlier
            remaining_sum = total_sum - num
            
            # Check if remaining_sum is even
            if remaining_sum % 2 != 0:
                continue
            
            target = remaining_sum // 2
            
            # Check if target exists in the remaining elements
            if target in freq and (target != num or freq[target] > 1):
                max_outlier = max(max_outlier, num)
        
        return max_outlier