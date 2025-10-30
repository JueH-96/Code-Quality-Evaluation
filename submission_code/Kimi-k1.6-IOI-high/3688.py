class Solution:
    def maxSubarraySum(self, nums: List[int]) -> int:
        # Compute the maximum subarray sum of the original array
        original_max = -float('inf')
        current_sum = -float('inf')
        for num in nums:
            if current_sum == -float('inf'):
                current_sum = num
            else:
                current_sum = max(num, current_sum + num)
            if current_sum > original_max:
                original_max = current_sum
        
        max_sum = original_max
        from collections import defaultdict
        counts = defaultdict(int)
        for num in nums:
            counts[num] += 1
        
        # Iterate through unique elements and compute max sum after removing each
        unique_elements = list(counts.keys())
        n = len(nums)
        for x in unique_elements:
            if counts[x] == n:
                continue  # Removing all x's would make the array empty
            # Compute max subarray sum when x is removed
            current_max = -float('inf')
            temp_sum = -float('inf')
            for num in nums:
                if num == x:
                    continue  # Skip this element
                if temp_sum == -float('inf'):
                    temp_sum = num
                else:
                    temp_sum = max(num, temp_sum + num)
                if temp_sum > current_max:
                    current_max = temp_sum
            # Update the overall maximum sum
            if current_max > max_sum:
                max_sum = current_max
        
        return max_sum