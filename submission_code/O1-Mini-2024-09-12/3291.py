class Solution:
    def canSortArray(self, nums: List[int]) -> bool:
        def count_set_bits(x):
            return bin(x).count('1')
        
        n = len(nums)
        nums_copy = nums[:]
        sorted_nums = sorted(nums_copy)
        swapped = True
        while swapped:
            swapped = False
            for i in range(n-1):
                if nums_copy[i] > nums_copy[i+1] and count_set_bits(nums_copy[i]) == count_set_bits(nums_copy[i+1]):
                    nums_copy[i], nums_copy[i+1] = nums_copy[i+1], nums_copy[i]
                    swapped = True
        return nums_copy == sorted_nums