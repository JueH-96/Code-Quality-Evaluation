class Solution:
    def isGood(self, nums: List[int]) -> bool:
        max_num = max(nums)
        if nums.count(max_num) != 2:
            return False
        for i in range(1, max_num):
            if nums.count(i) != 1:
                return False
        return True