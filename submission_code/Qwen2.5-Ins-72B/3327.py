from typing import List
import bisect

class Solution:
    def minimumMoves(self, nums: List[int], k: int, maxChanges: int) -> int:
        ones = [i for i, num in enumerate(nums) if num == 1]
        n = len(ones)
        
        def cost(i, j):
            mid = (i + j) // 2
            return sum(abs(ones[mid] - ones[x]) for x in range(i, j + 1))
        
        min_moves = float('inf')
        
        for i in range(n + 1):
            for j in range(max(0, i - maxChanges), min(n, i + maxChanges) + 1):
                if j - i + maxChanges >= k:
                    moves = cost(i, j - 1) + (k - (j - i)) * 2
                    min_moves = min(min_moves, moves)
        
        return min_moves