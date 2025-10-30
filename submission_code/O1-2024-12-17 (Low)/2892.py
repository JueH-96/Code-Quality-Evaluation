class Solution:
    def isGood(self, nums: List[int]) -> bool:
        # Find the maximum element in nums
        n = max(nums)
        
        # The length of base[n] should be n+1.
        if len(nums) != n + 1:
            return False
        
        # Construct the base array: [1, 2, ..., n-1, n, n]
        base_array = list(range(1, n)) + [n, n]
        
        # If the sorted version of nums matches sorted base_array, then it's a permutation
        return sorted(nums) == sorted(base_array)