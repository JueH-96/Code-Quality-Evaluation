class Solution:
    def semiOrderedPermutation(self, nums: List[int]) -> int:
        n = len(nums)
        index_one = -1
        index_n = -1
        for i in range(n):
            if nums[i] == 1:
                index_one = i
            if nums[i] == n:
                index_n = i

        swaps = 0
        swaps += index_one
        swaps += (n - 1 - index_n)

        if index_one > index_n:
            swaps -= 1
        return swaps