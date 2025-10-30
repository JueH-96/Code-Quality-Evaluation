class Solution:
    def minimumMoves(self, nums: List[int], k: int, maxChanges: int) -> int:
        n = len(nums)
        ones_positions = [i for i in range(n) if nums[i] == 1]
        ones_count = len(ones_positions)

        if ones_count >= k:
            return k  # If there are enough ones, just pick them up

        # Calculate how many zeros we need to convert to ones
        needed_ones = k - ones_count
        if needed_ones > maxChanges:
            return float('inf')  # Not enough changes allowed

        # We will use a sliding window to find the minimum moves
        min_moves = float('inf')

        for start in range(ones_count):
            end = start + k - 1
            if end >= ones_count:
                break
            
            # Calculate the number of moves needed to collect k ones
            left_most = ones_positions[start]
            right_most = ones_positions[end]
            total_moves = (right_most - left_most + 1) - (end - start + 1)  # Total slots - filled slots
            
            # We can convert some zeros to ones
            if total_moves <= maxChanges:
                min_moves = min(min_moves, total_moves + k)

        return min_moves