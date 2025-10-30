class Solution:
    def canSortArray(self, nums: List[int]) -> bool:
        def count_bits(n):
            count = 0
            while n:
                count += n & 1
                n >>= 1
            return count

        def is_sorted(nums):
            return all(nums[i] <= nums[i + 1] for i in range(len(nums) - 1))

        def can_sort(nums, i):
            if i == len(nums):
                return is_sorted(nums)
            for j in range(i, len(nums)):
                nums[i], nums[j] = nums[j], nums[i]
                if can_sort(nums, i + 1):
                    return True
                nums[i], nums[j] = nums[j], nums[i]
            return False

        return can_sort(nums, 0)