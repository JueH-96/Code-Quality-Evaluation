class Solution:
    def minSizeSubarray(self, nums, target):
        n = len(nums)
        nums = nums * 2
        prefix_sum = [0] * (2 * n + 1)
        for i in range(1, 2 * n + 1):
            prefix_sum[i] = prefix_sum[i - 1] + nums[i - 1]
        min_length = float('inf')
        deque = []
        for i in range(2 * n + 1):
            while deque and prefix_sum[i] - prefix_sum[deque[0]] >= target:
                min_length = min(min_length, i - deque[0])
                deque.pop(0)
            while deque and prefix_sum[i] <= prefix_sum[deque[-1]]:
                deque.pop()
            deque.append(i)
        return min_length if min_length <= n else -1