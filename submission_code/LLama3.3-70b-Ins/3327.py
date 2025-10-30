from typing import List

class Solution:
    def minimumMoves(self, nums: List[int], k: int, maxChanges: int) -> int:
        n = len(nums)
        ones = [i for i, x in enumerate(nums) if x == 1]
        k_ones = ones[:k]
        
        def check(mid):
            changes = maxChanges
            for i in range(1, k):
                if k_ones[i] - k_ones[i - 1] > 1:
                    if changes == 0:
                        return False
                    changes -= 1
            return True
        
        left, right = 0, n
        while left < right:
            mid = (left + right) // 2
            if check(mid):
                right = mid
            else:
                left = mid + 1
        
        return left