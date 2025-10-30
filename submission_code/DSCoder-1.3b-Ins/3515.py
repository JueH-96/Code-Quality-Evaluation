class Solution:
    def canAliceWin(self, nums: List[int]) -> bool:
        memo = {}
        def helper(nums):
            s = ''.join(map(str, nums))
            if s in memo:
                return memo[s]
            for i in range(len(nums)):
                for j in range(i + 1, len(nums)):
                    if nums[i] == nums[j]:
                        new_nums = nums[:i] + nums[i+1:j] + nums[j+1:]
                        if not helper(new_nums):
                            memo[s] = True
                            return True
            memo[s] = False
            return False
        return helper(nums)