from sortedcontainers import SortedList

class Solution:
    def lexicographicallySmallestArray(self, nums: List[int], limit: int) -> List[int]:
        n = len(nums)
        sorted_nums = SortedList(nums)
        min_nums = sorted_nums[::-1]
        l = 0
        for r in range(n):
            while min_nums[r] - sorted_nums[l] > limit:
                l += 1
            nums[r] = sorted_nums[l]
            sorted_nums.remove(sorted_nums[l])
            min_nums = sorted_nums[::-1]
        return nums