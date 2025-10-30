from typing import List
import heapq

class Solution:
    def minimumMoves(self, nums: List[int], k: int, maxChanges: int) -> int:
        n = len(nums)
        ones_positions = [i for i, num in enumerate(nums) if num == 1]

        if len(ones_positions) >= k:
            ones_positions = ones_positions[:k]
            return ones_positions[-1] - ones_positions[0] + 1 - k

        min_heap = []
        for pos in ones_positions:
            heapq.heappush(min_heap, (pos, pos))

        current_sum = sum(ones_positions)
        maxChanges -= (k - len(ones_positions))

        for i in range(n):
            if nums[i] == 0 and maxChanges > 0:
                nums[i] = 1
                maxChanges -= 1
                heapq.heappush(min_heap, (i, i))
                current_sum += i
                if len(min_heap) > k:
                    _, remove_pos = heapq.heappop(min_heap)
                    current_sum -= remove_pos

        positions = [pos for _, pos in min_heap]
        positions.sort()
        return positions[-1] - positions[0] + 1 - k

# Example usage:
# solution = Solution()
# print(solution.minimumMoves([1,1,0,0,0,1,1,0,0,1], 3, 1))  # Output: 3
# print(solution.minimumMoves([0,0,0,0], 2, 3))  # Output: 4