class Solution:
    def minAbsoluteDifference(self, nums: List[int], x: int) -> int:
        import bisect
        
        n = len(nums)
        
        # Special case: when x = 0, any two distinct indices work
        if x == 0:
            sorted_nums = sorted(nums)
            min_diff = float('inf')
            for i in range(1, n):
                min_diff = min(min_diff, sorted_nums[i] - sorted_nums[i-1])
            return min_diff
        
        min_diff = float('inf')
        sorted_list = []
        
        # For each position i from x to n-1
        for i in range(x, n):
            # Add the element that is exactly x positions before
            bisect.insort(sorted_list, nums[i - x])
            
            # Find the closest value in sorted_list to nums[i]
            idx = bisect.bisect_left(sorted_list, nums[i])
            
            # Check the value at idx (first element >= nums[i])
            if idx < len(sorted_list):
                min_diff = min(min_diff, abs(nums[i] - sorted_list[idx]))
            
            # Check the value at idx-1 (largest element < nums[i])
            if idx > 0:
                min_diff = min(min_diff, abs(nums[i] - sorted_list[idx - 1]))
        
        return min_diff