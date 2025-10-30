class Solution:
    def minimumMoves(self, nums, k, maxChanges):
        n = len(nums)
        ones = [i for i in range(n) if nums[i] == 1]
        ones = [0] + ones + [n] * k
        prefix = [0] * (len(ones))
        for i in range(1, len(ones)):
            prefix[i] = prefix[i - 1] + ones[i]
        res = float('inf')
        for i in range(k, len(ones) - k):
            left = i - k
            right = i + k
            total = prefix[right] - prefix[left]
            mid = (left + right) // 2
            moves = total - ones[mid] * k
            res = min(res, moves)
        return res - maxChanges