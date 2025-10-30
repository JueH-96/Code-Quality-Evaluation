class Solution:
    def maxOperations(self, nums: List[int]) -> int:
        if len(nums) < 2:
            return 0
        
        # The score we want each operation's sum to match
        required_sum = nums[0] + nums[1]
        
        count = 0
        i = 0
        while i < len(nums) - 1:
            # Check if the current first pair matches the required sum
            if nums[i] + nums[i + 1] == required_sum:
                count += 1
                i += 2  # Move past the pair we just removed
            else:
                break
        
        return count