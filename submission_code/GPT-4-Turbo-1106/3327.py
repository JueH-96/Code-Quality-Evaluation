class Solution:
    def minimumMoves(self, nums: list[int], k: int, maxChanges: int) -> int:
        # Helper function to calculate the minimum moves within a window
        def min_moves_in_window(window: list[int], ones: int) -> int:
            left, right, zero_count, min_moves = 0, 0, 0, float('inf')
            while right < len(window):
                if window[right] == 0:
                    zero_count += 1
                while zero_count > maxChanges:
                    if window[left] == 0:
                        zero_count -= 1
                    left += 1
                min_moves = min(min_moves, right - left + 1 - ones)
                right += 1
            return min_moves

        # Find the indices of all ones in the array
        ones_indices = [i for i, num in enumerate(nums) if num == 1]
        total_ones = len(ones_indices)

        # If we already have enough ones, no moves are needed
        if total_ones >= k:
            return 0

        # Initialize the minimum moves to a large number
        min_moves = float('inf')

        # Slide a window of size k across the ones_indices
        for i in range(total_ones - k + 1):
            window = ones_indices[i:i + k]
            # Calculate the number of moves needed to bring k ones together
            moves_needed = min_moves_in_window(window, k)
            # Update the minimum moves
            min_moves = min(min_moves, moves_needed)

        return min_moves

# Example usage:
# sol = Solution()
# print(sol.minimumMoves([1,1,0,0,0,1,1,0,0,1], 3, 1))  # Output: 3
# print(sol.minimumMoves([0,0,0,0], 2, 3))  # Output: 4