class Solution:
    def maxOperations(self, nums: List[int]) -> int:
        from collections import Counter
        
        if len(nums) < 2:
            return 0
        
        # Count the frequency of each sum of pairs
        sum_count = Counter()
        for i in range(0, len(nums) - 1, 2):
            pair_sum = nums[i] + nums[i + 1]
            sum_count[pair_sum] += 1
        
        # If there's an odd number of elements, consider the last element with the first for one more pair
        if len(nums) % 2 == 1:
            pair_sum = nums[-1] + nums[0]
            sum_count[pair_sum] += 1
        
        # The maximum operations with the same score is the highest frequency of any sum
        return max(sum_count.values())