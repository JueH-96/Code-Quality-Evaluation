class Solution:
    def canSplitArray(self, nums: List[int], m: int) -> bool:
        n = len(nums)
        if n <= 1:
            return True
        for i in range(1, n):
            left_arr = nums[:i]
            right_arr = nums[i:]
            is_left_valid = (len(left_arr) == 1) or (sum(left_arr) >= m)
            is_right_valid = (len(right_arr) == 1) or (sum(right_arr) >= m)
            if is_left_valid and is_right_valid:
                return True
        return False