class Solution:
    def minimumMoves(self, nums: List[int], k: int, maxChanges: int) -> int:
        n = len(nums)
        
        # Find the indices of the 1s
        ones = [i for i in range(n) if nums[i] == 1]
        
        # If there are less than k 1s, return -1
        if len(ones) < k:
            return -1
        
        # Initialize the minimum moves required
        min_moves = float('inf')
        
        # Try all possible starting indices
        for start in range(n - k + 1):
            # Count the number of moves required
            moves = 0
            changes = 0
            
            # Pick up the first k ones
            for i in range(start, start + k):
                if nums[i] == 0:
                    if changes < maxChanges:
                        changes += 1
                    else:
                        # Swap the adjacent 0 and 1
                        for j in range(i - 1, -1, -1):
                            if nums[j] == 1:
                                nums[j], nums[i] = 0, 1
                                moves += 1
                                break
                else:
                    nums[i] = 0
                    moves += 1
            
            # Update the minimum moves required
            min_moves = min(min_moves, moves)
            
            # Reset the array
            for i in range(start, start + k):
                if nums[i] == 0:
                    nums[i] = 1
        
        return min_moves