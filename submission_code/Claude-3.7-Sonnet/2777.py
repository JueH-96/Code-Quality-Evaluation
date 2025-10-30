class Solution:
    def distinctDifferenceArray(self, nums: List[int]) -> List[int]:
        n = len(nums)
        
        # Precompute the number of distinct elements in the prefix
        prefix_count = []
        prefix_set = set()
        for num in nums:
            prefix_set.add(num)
            prefix_count.append(len(prefix_set))
        
        # Precompute the number of distinct elements in the suffix
        suffix_count = [0] * n
        suffix_set = set()
        for i in range(n - 1, -1, -1):
            suffix_count[i] = len(suffix_set)
            suffix_set.add(nums[i])
        
        # Calculate diff
        return [prefix_count[i] - suffix_count[i] for i in range(n)]