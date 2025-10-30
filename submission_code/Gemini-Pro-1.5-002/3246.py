class Solution:
    def hasTrailingZeros(self, nums: List[int]) -> bool:
        n = len(nums)
        for i in range(n):
            for j in range(i + 1, n):
                or_val = nums[i] | nums[j]
                if or_val & 1 == 0:
                    return True
        return False