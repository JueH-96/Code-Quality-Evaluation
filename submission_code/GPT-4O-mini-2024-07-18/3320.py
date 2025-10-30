class Solution:
    def maxOperations(self, nums: List[int]) -> int:
        from collections import defaultdict
        
        score_count = defaultdict(int)
        max_operations = 0
        
        while len(nums) > 1:
            score = nums[0] + nums[1]
            score_count[score] += 1
            nums = nums[2:]  # Remove the first two elements
            
        for count in score_count.values():
            max_operations = max(max_operations, count)
        
        return max_operations