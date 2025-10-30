class Solution:
    def semiOrderedPermutation(self, nums: List[int]) -> int:
        n = len(nums)
        index_1 = -1
        index_n = -1
        for i in range(n):
            if nums[i] == 1:
                index_1 = i
            if nums[i] == n:
                index_n = i
        swaps1 = index_1
        swaps2 = 0
        if index_1 < index_n:
            swaps2 = n - 1 - index_n
        else:
            swaps2 = n - 2 - index_n
        return swaps1 + swaps2