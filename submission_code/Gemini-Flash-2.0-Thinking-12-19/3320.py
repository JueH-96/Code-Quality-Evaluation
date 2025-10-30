class Solution:
    def maxOperations(self, nums: List[int]) -> int:
        count = 0
        if len(nums) < 2:
            return 0
        
        initial_score = nums[0] + nums[1]
        nums = nums[2:]
        count += 1
        
        while len(nums) >= 2:
            current_score = nums[0] + nums[1]
            if current_score == initial_score:
                nums = nums[2:]
                count += 1
            else:
                break
                
        return count