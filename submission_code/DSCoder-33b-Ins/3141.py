class Solution:
    def minSizeSubarray(self, nums: List[int], target: int) -> int:
        prefix_sum = [0] * (2 * len(nums) + 1)
        for i in range(2 * len(nums)):
            prefix_sum[i + 1] = prefix_sum[i] + nums[i % len(nums)]

        min_length = float('inf')
        deque = collections.deque([(0, 0)])
        for right in range(1, len(prefix_sum)):
            while deque and prefix_sum[right] - prefix_sum[deque[0][0]] >= target:
                min_length = min(min_length, right - deque.popleft()[0])
            while deque and prefix_sum[deque[-1][0]] >= prefix_sum[right]:
                deque.pop()
            deque.append((right, prefix_sum[right]))

        return min_length if min_length != float('inf') else -1