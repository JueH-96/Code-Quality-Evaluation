class Solution:
    def semiOrderedPermutation(self, nums: List[int]) -> int:
        n = len(nums)
        # find index of value 1 and index of value n (largest)
        idx1 = nums.index(1)
        idxn = nums.index(n)
        
        # If the element 1 is on the right of n, then when moving 1 left,
        # 1 will effectively "pass" n, reducing the total swap count by 1.
        if idx1 > idxn:
            return idx1 + (n - 1 - idxn) - 1
        else:
            return idx1 + (n - 1 - idxn)