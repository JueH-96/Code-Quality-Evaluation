class Solution:
    def canSortArray(self, nums: List[int]) -> bool:
        import copy
        from typing import List

        def count_set_bits(n):
            count = 0
            while n:
                count += n & 1
                n >>= 1
            return count

        original = copy.deepcopy(nums)
        sorted_nums = sorted(nums)
        n = len(nums)

        # Perform up to n-1 passes
        for i in range(n - 1):
            for j in range(n - 1 - i):
                if nums[j] > nums[j + 1]:
                    if count_set_bits(nums[j]) == count_set_bits(nums[j + 1]):
                        nums[j], nums[j + 1] = nums[j + 1], nums[j]

        # Check if the array is sorted
        return nums == sorted_nums