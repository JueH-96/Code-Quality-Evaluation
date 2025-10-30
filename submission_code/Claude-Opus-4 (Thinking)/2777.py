class Solution:
    def distinctDifferenceArray(self, nums: List[int]) -> List[int]:
        n = len(nums)
        
        # Pre-compute distinct count for all suffixes
        # suffix_count[i] = number of distinct elements in nums[i:]
        suffix_count = [0] * (n + 1)
        seen = set()
        
        # Build suffix counts from right to left
        for i in range(n-1, -1, -1):
            seen.add(nums[i])
            suffix_count[i] = len(seen)
        
        # Build the result array
        diff = []
        prefix_seen = set()
        
        for i in range(n):
            prefix_seen.add(nums[i])
            # For diff[i], we need distinct count in nums[i+1:]
            # which is stored in suffix_count[i+1]
            diff.append(len(prefix_seen) - suffix_count[i+1])
        
        return diff