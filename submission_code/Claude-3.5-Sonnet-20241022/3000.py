class Solution:
    def minAbsoluteDifference(self, nums: List[int], x: int) -> int:
        if x == 0:
            return 0
            
        n = len(nums)
        min_diff = float('inf')
        
        # Use sorted list to efficiently find closest values
        from sortedcontainers import SortedList
        sorted_nums = SortedList()
        
        for i in range(x, n):
            # Add number from left side to sorted list
            sorted_nums.add(nums[i-x])
            
            # Find closest values to current number
            curr = nums[i]
            idx = sorted_nums.bisect_left(curr)
            
            # Check value equal or greater
            if idx < len(sorted_nums):
                min_diff = min(min_diff, abs(sorted_nums[idx] - curr))
            
            # Check value less
            if idx > 0:
                min_diff = min(min_diff, abs(sorted_nums[idx-1] - curr))
                
        return min_diff