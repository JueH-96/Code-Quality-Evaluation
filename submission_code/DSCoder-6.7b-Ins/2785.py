class Solution:
    def semiOrderedPermutation(self, nums: List[int]) -> int:
        n = len(nums)
        index_dict = {num: i for i, num in enumerate(nums)}
        ops = 0
        if nums[0] != 1:
            one_index = index_dict[1]
            nums[one_index], nums[0] = nums[0], nums[one_index]
            ops += 1
        for i in range(1, n):
            if nums[i] != i + 1:
                next_index = index_dict[i + 1]
                nums[next_index], nums[i] = nums[i], nums[next_index]
                ops += 1
        return ops