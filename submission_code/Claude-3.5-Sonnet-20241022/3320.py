class Solution:
    def maxOperations(self, nums: List[int]) -> int:
        n = len(nums)
        if n < 2:
            return 0
            
        # Get first operation score
        first_score = nums[0] + nums[1]
        
        # Count max operations with same score
        count = 1
        i = 2
        
        while i < n-1:
            curr_score = nums[i] + nums[i+1]
            if curr_score != first_score:
                break
            count += 1
            i += 2
            
        return count