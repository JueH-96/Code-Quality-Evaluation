class Solution:
    def minimumSeconds(self, nums: List[int]) -> int:
        n = len(nums)
        nums = nums + nums
        min_sec = float('inf')
        for num in set(nums):
            left, right = [], []
            for i in range(len(nums)):
                if nums[i] == num:
                    left.append(i)
                else:
                    right.append(i)
            left = [0] + left + [len(nums)]
            right = [0] + right + [len(nums)]
            changes = [(right[i+1]-left[i], left[i+1]-right[i]) for i in range(len(left)-1)]
            min_changes = min(min(changes))
            min_sec = min(min_sec, min_changes//2)
        return min_sec