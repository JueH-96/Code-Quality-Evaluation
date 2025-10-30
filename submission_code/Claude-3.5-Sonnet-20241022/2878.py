class Solution:
    def checkArray(self, nums: List[int], k: int) -> bool:
        n = len(nums)
        if k == 1:
            return True
            
        # Track running decrements
        decrements = [0] * (n + 1)
        curr_decrement = 0
        
        for i in range(n):
            # Add decrement from k positions ago
            if i > 0:
                curr_decrement -= decrements[i]
            
            # Add current position's decrement
            nums[i] -= curr_decrement
            
            if nums[i] < 0:
                return False
                
            if nums[i] > 0:
                if i + k > n:
                    return False
                # Need to decrement this and next k-1 elements
                decrements[i + k] = nums[i]
                curr_decrement += nums[i]
                nums[i] = 0
                
        return True