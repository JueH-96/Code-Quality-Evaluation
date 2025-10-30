class Solution:
    def maxOperations(self, nums: List[int]) -> int:
        from collections import defaultdict
        
        # Dictionary to store the frequency of each possible score
        score_count = defaultdict(int)
        
        # Iterate over the array in pairs
        for i in range(0, len(nums) - 1, 2):
            score = nums[i] + nums[i + 1]
            score_count[score] += 1
        
        # If the length of nums is odd, we need to consider the last element
        # which cannot form a pair and thus should be ignored.
        if len(nums) % 2 == 1:
            # Check the last element with the second last element
            score = nums[-2] + nums[-1]
            score_count[score] += 1
        
        # Find the maximum number of operations with the same score
        max_operations = max(score_count.values())
        
        return max_operations