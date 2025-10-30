class Solution:
    def minimumMoves(self, nums: List[int], k: int, maxChanges: int) -> int:
        n = len(nums)
        
        # Calculate prefix sums of ones
        prefix_ones = [0] * (n + 1)
        for i in range(n):
            prefix_ones[i + 1] = prefix_ones[i] + nums[i]
        
        # Function to calculate the number of ones in a subarray [l, r]
        def ones_in_range(l, r):
            return prefix_ones[r + 1] - prefix_ones[l]
        
        # Initialize the minimum moves to a large number
        min_moves = float('inf')
        
        # Try every possible starting position for Alice
        for start in range(n):
            # Calculate the number of ones Alice can pick up from this position
            ones_picked = 0
            moves = 0
            changes_left = maxChanges
            current_position = start
            
            while ones_picked < k and current_position < n:
                if nums[current_position] == 1:
                    ones_picked += 1
                else:
                    if changes_left > 0:
                        changes_left -= 1
                        ones_picked += 1
                    else:
                        # Find the nearest 1 to swap with
                        swap_position = current_position
                        while swap_position < n and nums[swap_position] == 0:
                            swap_position += 1
                        if swap_position < n:
                            moves += swap_position - current_position
                            current_position = swap_position
                            ones_picked += 1
                        else:
                            break
                current_position += 1
            
            if ones_picked >= k:
                min_moves = min(min_moves, moves)
        
        return min_moves