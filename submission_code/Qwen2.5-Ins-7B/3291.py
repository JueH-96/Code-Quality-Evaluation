class Solution:
    def canSortArray(self, nums: List[int]) -> bool:
        n = len(nums)
        for i in range(n - 1):
            while i < n - 1 and nums[i] > nums[i + 1]:
                if bin(nums[i]).count('1') != bin(nums[i + 1]).count('1'):
                    return False
                nums[i], nums[i + 1] = nums[i + 1], nums[i]
                i -= 1
                i = max(i, 0)
            i += 1
        return True