class Solution:
    def semiOrderedPermutation(self, nums: List[int]) -> int:
        n = len(nums)
        if nums[0] == 1 and nums[-1] == n:
            return 0
        
        count = 0
        
        #Move 1 to the beginning
        index_1 = nums.index(1)
        count += index_1
        nums = nums[index_1:] + nums[:index_1]

        #Move n to the end
        index_n = nums.index(n)
        count += (n - 1 - index_n)
        nums = nums[:index_n] + nums[index_n+1:] + [nums[index_n]]

        return count