class Solution:
    def minSizeSubarray(self, nums: List[int], target: int) -> int:
        total = sum(nums)
        n = len(nums)
        min_len = float('inf')
        
        # If the target is larger than the total sum of the array, we need to consider multiple copies
        # of the array. The number of full copies needed is target // total.
        # The remaining sum to be achieved is target % total.
        # However, if target is a multiple of total, then the entire array is a candidate.
        
        # First, handle the case where the target is a multiple of the total sum
        if target % total == 0:
            return (target // total) * n
        
        # Otherwise, we need to find the smallest subarray that sums to (target % total) in the array
        # and then add the necessary full copies.
        remainder = target % total
        if remainder == 0:
            return (target // total) * n
        
        # Now, find the smallest subarray in nums that sums to remainder
        # We can use the sliding window approach
        left = 0
        current_sum = 0
        min_sub_len = float('inf')
        
        for right in range(n * 2):
            current_sum += nums[right % n]
            while current_sum > remainder and left <= right:
                current_sum -= nums[left % n]
                left += 1
            if current_sum == remainder:
                min_sub_len = min(min_sub_len, right - left + 1)
        
        if min_sub_len == float('inf'):
            return -1
        
        # The total length is the number of full copies * n + the length of the subarray
        return (target // total) * n + min_sub_len