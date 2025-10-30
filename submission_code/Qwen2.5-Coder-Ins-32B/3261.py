from typing import List

class Solution:
    def minOrAfterOperations(self, nums: List[int], k: int) -> int:
        def canAchieve(target: int) -> bool:
            count = 0
            current_and = -1
            for num in nums:
                if current_and == -1:
                    current_and = num
                current_and &= num
                if current_and & target == 0:
                    current_and = -1
                    count += 1
                    if count > k:
                        return False
            return True

        low, high = 0, (1 << 30) - 1
        while low < high:
            mid = (low + high) // 2
            if canAchieve(mid):
                high = mid
            else:
                low = mid + 1
        return low