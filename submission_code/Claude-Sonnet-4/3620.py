class Solution:
    def maxDistinctElements(self, nums: List[int], k: int) -> int:
        nums.sort()
        
        last_assigned = float('-inf')
        count = 0
        
        for num in nums:
            # The range of possible values for this element
            min_val = num - k
            max_val = num + k
            
            # We want to assign the smallest value >= last_assigned + 1
            # that's also within the valid range [min_val, max_val]
            target = max(min_val, last_assigned + 1)
            
            # If target is within the valid range, we can assign it
            if target <= max_val:
                last_assigned = target
                count += 1
            # Otherwise, this element cannot contribute to distinct count
        
        return count