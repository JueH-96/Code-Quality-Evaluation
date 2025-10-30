class Solution:
    def distinctDifferenceArray(self, nums: List[int]) -> List[int]:
        n = len(nums)

        # Calculate the number of distinct elements in the prefix up to index i
        prefix_distinct = []
        seen_prefix = set()
        for i in range(n):
            seen_prefix.add(nums[i])
            prefix_distinct.append(len(seen_prefix))
        
        # Calculate the number of distinct elements in the suffix starting from index i
        suffix_distinct = [0] * n
        seen_suffix = set()
        for i in range(n - 1, -1, -1):
            seen_suffix.add(nums[i])
            suffix_distinct[i] = len(seen_suffix)
        
        # Construct the result based on the problem requirement
        diff = []
        for i in range(n):
            # Suffix for index i is from i+1 to n-1
            suffix_val = suffix_distinct[i + 1] if i + 1 < n else 0
            diff.append(prefix_distinct[i] - suffix_val)
        
        return diff