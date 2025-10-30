class Solution:
    def maxOperations(self, nums: List[int]) -> int:
        from collections import Counter
        
        # Count the frequency of each sum of pairs
        pairs_sum = Counter()
        for i in range(0, len(nums) - 1, 2):
            pairs_sum[nums[i] + nums[i + 1]] += 1
        
        # If there's an odd number of elements, consider the last element as a single pair
        if len(nums) % 2 == 1:
            pairs_sum[nums[-1]] += 1
        
        # Return the maximum frequency among the sums
        return max(pairs_sum.values())