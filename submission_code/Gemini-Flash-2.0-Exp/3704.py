class Solution:
    def countPartitions(self, nums: List[int]) -> int:
        n = len(nums)
        count = 0
        for i in range(n - 1):
            left_sum = sum(nums[:i+1])
            right_sum = sum(nums[i+1:])
            if (left_sum - right_sum) % 2 == 0:
                count += 1
        return count