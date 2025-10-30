class Solution:
    def minimumMoves(self, nums: List[int], k: int, maxChanges: int) -> int:
        n = len(nums)
        ones = [i for i in range(n) if nums[i] == 1]

        if len(ones) >= k:
            return k

        min_moves = float('inf')

        for i in range(len(ones) - k + 1):
            left = ones[i]
            right = ones[i + k - 1]
            zeros = right - left + 1 - k

            if zeros <= maxChanges:
                min_moves = min(min_moves, zeros)

        return min_moves if min_moves != float('inf') else -1