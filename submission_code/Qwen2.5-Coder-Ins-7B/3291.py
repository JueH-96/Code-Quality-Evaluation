class Solution:
    def canSortArray(self, nums: List[int]) -> bool:
        def count_set_bits(n):
            return bin(n).count('1')
        
        n = len(nums)
        for i in range(n):
            for j in range(i + 1, n):
                if count_set_bits(nums[i]) == count_set_bits(nums[j]):
                    nums[i], nums[j] = nums[j], nums[i]
        
        return nums == sorted(nums)