class Solution:
    def minAbsoluteDifference(self, nums: List[int], x: int) -> int:
        from sortedcontainers import SortedList
        
        n = len(nums)
        if x == 0:
            return 0
        
        min_diff = float('inf')
        sorted_list = SortedList()
        
        # Process each element
        for i in range(n):
            # Add elements that are at least x positions behind current index
            if i >= x:
                sorted_list.add(nums[i - x])
            
            # Find closest value in sorted_list to nums[i]
            if sorted_list:
                # Binary search for position where nums[i] would be inserted
                pos = sorted_list.bisect_left(nums[i])
                
                # Check the element at or after this position
                if pos < len(sorted_list):
                    min_diff = min(min_diff, abs(nums[i] - sorted_list[pos]))
                
                # Check the element before this position
                if pos > 0:
                    min_diff = min(min_diff, abs(nums[i] - sorted_list[pos - 1]))
        
        # Also process from right to left to ensure we check all valid pairs
        sorted_list = SortedList()
        for i in range(n - 1, -1, -1):
            # Add elements that are at least x positions ahead of current index
            if i + x < n:
                sorted_list.add(nums[i + x])
            
            # Find closest value in sorted_list to nums[i]
            if sorted_list:
                pos = sorted_list.bisect_left(nums[i])
                
                if pos < len(sorted_list):
                    min_diff = min(min_diff, abs(nums[i] - sorted_list[pos]))
                
                if pos > 0:
                    min_diff = min(min_diff, abs(nums[i] - sorted_list[pos - 1]))
        
        return min_diff