class Solution:
    def isGood(self, nums: List[int]) -> bool:
        from collections import Counter
        count = Counter(nums)
        max_num = max(nums)
        if len(nums) != max_num + 1:
            return False
        for i in range(1, max_num):
            if count.get(i, 0) != 1:
                return False
        if count.get(max_num, 0) != 2:
            return False
        return True