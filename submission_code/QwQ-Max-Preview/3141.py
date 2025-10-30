class Solution:
    def minSizeSubarray(self, nums: List[int], target: int) -> int:
        total = sum(nums)
        n = len(nums)
        if total == 0:
            return -1  # As per constraints, sum can't be zero
        
        k = target // total
        remainder = target % total
        
        if remainder == 0:
            return k * n
        
        # Create the concatenated array of nums + nums to handle wrapping around once
        concatenated = nums + nums
        min_len = float('inf')
        current_sum = 0
        left = 0
        
        for right in range(len(concatenated)):
            current_sum += concatenated[right]
            
            while current_sum > remainder and left <= right:
                current_sum -= concatenated[left]
                left += 1
            
            if current_sum == remainder:
                min_len = min(min_len, right - left + 1)
        
        if min_len == float('inf'):
            return -1
        else:
            return k * n + min_len