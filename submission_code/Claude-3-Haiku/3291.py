class Solution:
    def canSortArray(self, nums: List[int]) -> bool:
        def count_set_bits(num):
            count = 0
            while num:
                count += num & 1
                num >>= 1
            return count

        n = len(nums)
        for i in range(n - 1):
            for j in range(i + 1, n):
                if count_set_bits(nums[i]) == count_set_bits(nums[j]):
                    nums[i], nums[j] = nums[j], nums[i]
                    if nums == sorted(nums):
                        return True
        return False