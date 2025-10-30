class Solution:
    def isPossibleToSplit(self, nums: List[int]) -> bool:
        prefix = [0] * (len(nums) + 1)
        for i in range(len(nums)):
            prefix[i+1] = prefix[i] + nums[i]
        prefix_set = set()
        for i in range(len(nums) - 1):
            if nums[i] in prefix_set:
                continue
            prefix_set.add(nums[i])
            left = prefix[i+1] - nums[i]
            right = prefix[-1] - left
            if left <= right:
                return True
        return False