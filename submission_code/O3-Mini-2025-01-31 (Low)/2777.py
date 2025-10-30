class Solution:
    def distinctDifferenceArray(self, nums: List[int]) -> List[int]:
        n = len(nums)
        res = [0] * n
        
        # Precompute prefix distinct counts
        prefix_set = set()
        prefix_counts = []
        for num in nums:
            prefix_set.add(num)
            prefix_counts.append(len(prefix_set))
        
        # Precompute suffix distinct counts
        suffix_set = set()
        suffix_counts = [0] * n
        # suffix for last element is empty so distinct count = 0
        for i in range(n - 1, -1, -1):
            # start adding from current index's right neighbor
            if i + 1 < n:
                suffix_set.add(nums[i+1])
            suffix_counts[i] = len(suffix_set)
            
        for i in range(n):
            res[i] = prefix_counts[i] - suffix_counts[i]
        return res