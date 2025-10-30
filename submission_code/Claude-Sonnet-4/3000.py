class Solution:
    def minAbsoluteDifference(self, nums: List[int], x: int) -> int:
        if x == 0:
            # If x is 0, we can compare any two elements, including the same element
            return 0
        
        n = len(nums)
        min_diff = float('inf')
        
        # We'll use a list to maintain sorted elements that are at least x positions away
        from bisect import bisect_left, insort
        
        valid_elements = []
        
        for i in range(n):
            # Add elements that are now at least x positions away from current position
            # Elements at positions <= i - x are valid
            if i >= x:
                insort(valid_elements, nums[i - x])
            
            # Also check elements at positions >= i + x (but we need to look ahead)
            # Actually, let's handle this differently - we'll process from both directions
            
            if valid_elements:
                # Find the closest element(s) to nums[i] in valid_elements
                target = nums[i]
                
                # Find insertion point
                pos = bisect_left(valid_elements, target)
                
                # Check element at pos (if exists) - this is >= target
                if pos < len(valid_elements):
                    min_diff = min(min_diff, abs(valid_elements[pos] - target))
                
                # Check element at pos-1 (if exists) - this is < target
                if pos > 0:
                    min_diff = min(min_diff, abs(valid_elements[pos-1] - target))
        
        # We also need to check in the other direction
        valid_elements = []
        for i in range(n-1, -1, -1):
            # Add elements that are now at least x positions away from current position
            if i + x < n:
                insort(valid_elements, nums[i + x])
            
            if valid_elements:
                target = nums[i]
                pos = bisect_left(valid_elements, target)
                
                if pos < len(valid_elements):
                    min_diff = min(min_diff, abs(valid_elements[pos] - target))
                
                if pos > 0:
                    min_diff = min(min_diff, abs(valid_elements[pos-1] - target))
        
        return min_diff