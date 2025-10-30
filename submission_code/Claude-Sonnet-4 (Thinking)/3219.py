class Solution:
    def lexicographicallySmallestArray(self, nums: List[int], limit: int) -> List[int]:
        n = len(nums)
        
        # Create a list of (value, index) and sort by value
        indexed_nums = [(nums[i], i) for i in range(n)]
        indexed_nums.sort()
        
        # Group elements that can be swapped
        groups = []
        for val, idx in indexed_nums:
            if not groups or val - groups[-1][-1][0] > limit:
                groups.append([(val, idx)])
            else:
                groups[-1].append((val, idx))
        
        # For each group, assign values optimally
        result = [0] * n
        for group in groups:
            values = [val for val, idx in group]
            indices = [idx for val, idx in group]
            indices.sort()
            for i, idx in enumerate(indices):
                result[idx] = values[i]
        
        return result