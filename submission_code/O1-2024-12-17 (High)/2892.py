class Solution:
    def isGood(self, nums: List[int]) -> bool:
        n = max(nums)
        # The length of a valid base[n] array must be n + 1
        if len(nums) != n + 1:
            return False
        
        # Construct the expected base[n] array
        expected = list(range(1, n)) + [n, n]
        
        # Compare the sorted input with the expected array
        return sorted(nums) == expected