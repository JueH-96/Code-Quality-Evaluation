from typing import List

class Solution:
    def minimumMoves(self, nums: List[int], k: int, maxChanges: int) -> int:
        ones = [i for i, num in enumerate(nums) if num == 1]
        n = len(ones)
        if n == 0:
            return k * 2 - (k <= maxChanges)
        
        prefix_sum = [0] * (n + 1)
        for i in range(n):
            prefix_sum[i + 1] = prefix_sum[i] + ones[i]
        
        min_moves = float('inf')
        for i in range(k - 1, n):
            mid = i - (k // 2)
            moves = (ones[mid] - ones[i - k + 1]) + (ones[i] - ones[mid]) - (prefix_sum[mid + 1] - prefix_sum[i - k + 1]) - (prefix_sum[i + 1] - prefix_sum[mid + 1])
            moves += max(0, k - (i - (i - k + 1) + 1)) * 2
            moves -= min(maxChanges, max(0, k - (i - (i - k + 1) + 1))) * 2
            min_moves = min(min_moves, moves)
        
        return min_moves