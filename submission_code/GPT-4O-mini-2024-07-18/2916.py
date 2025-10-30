class Solution:
    def canSplitArray(self, nums: List[int], m: int) -> bool:
        n = len(nums)
        
        # If the array has only one element, we cannot split it
        if n == 1:
            return False
        
        # We will use a stack to keep track of the current segments
        stack = []
        
        # Start with the whole array as a single segment
        stack.append(sum(nums))
        
        while stack:
            current_sum = stack.pop()
            
            # If the current sum is less than m and we have more than one element
            if current_sum < m and len(stack) == 0:
                return False
            
            # If we can split the current segment
            if current_sum >= m:
                # We can split it into two parts
                # We need to check if we can keep splitting until we have n segments
                if len(stack) + 1 < n:
                    # We can keep splitting
                    stack.append(current_sum // 2)  # Simulate a split
                    stack.append(current_sum - current_sum // 2)  # The other part
                else:
                    # We have enough segments
                    return True
            
            # If we have only one element left, we cannot split further
            if len(stack) == 0 and current_sum < m:
                return False
        
        return len(stack) == n