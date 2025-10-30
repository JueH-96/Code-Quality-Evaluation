class Solution:
    def semiOrderedPermutation(self, nums: List[int]) -> int:
        n = len(nums)
        one_index = nums.index(1)
        n_index = nums.index(n)
        
        ans = one_index + (n - 1 - n_index)
        
        if one_index > n_index:
            ans -= 1
            
        return ans