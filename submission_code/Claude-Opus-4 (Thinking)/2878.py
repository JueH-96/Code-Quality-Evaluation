class Solution:
    def checkArray(self, nums: List[int], k: int) -> bool:
        n = len(nums)
        curr_decrease = 0
        operations = [0] * n
        
        for i in range(n):
            # Remove effect of operations that have ended
            if i >= k:
                curr_decrease -= operations[i-k]
            
            # Current value after applying all decreases
            curr_val = nums[i] - curr_decrease
            
            if curr_val < 0:
                return False
            
            if curr_val > 0:
                if i + k > n:  # Can't start new operations
                    return False
                # Start curr_val operations at position i
                operations[i] = curr_val
                curr_decrease += curr_val
        
        return True