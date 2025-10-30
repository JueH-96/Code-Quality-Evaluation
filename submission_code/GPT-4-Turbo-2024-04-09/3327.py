class Solution:
    def minimumMoves(self, nums: List[int], k: int, maxChanges: int) -> int:
        from collections import deque
        
        # Collect all the indices of 1s in the array
        ones_indices = [i for i, value in enumerate(nums) if value == 1]
        n = len(ones_indices)
        
        # If k is 1, the answer is always 0 because Alice can start at any 1 and pick it up without any moves
        if k == 1:
            return 0
        
        # Sliding window to find the minimum cost to collect k ones
        min_moves = float('inf')
        current_changes = 0
        queue = deque()
        
        # We use a sliding window over the indices of ones to find the optimal position
        for i in range(n):
            # Add the new index to the queue
            queue.append(ones_indices[i])
            
            # If the window has more than k elements, remove the oldest one
            if len(queue) > k:
                queue.popleft()
            
            # When we have exactly k elements in the queue, calculate the cost
            if len(queue) == k:
                # The median index should be the target position for Alice to minimize moves
                # For k elements, the median index in sorted order is at k//2 (0-based index)
                median_index = queue[k//2]
                
                # Calculate the total moves required to align all k ones around the median index
                # We consider the median index as the center and calculate the cost to bring all other 1s to consecutive positions
                moves = 0
                for j, idx in enumerate(queue):
                    # Calculate the distance to the position it needs to be
                    # The target positions are median_index - k//2 + j
                    moves += abs(idx - (median_index - k//2 + j))
                
                # Update the minimum moves found so far
                min_moves = min(min_moves, moves)
        
        # Since we can use up to maxChanges to turn 0s into 1s, we need to subtract the minimum of maxChanges and (k-1)
        # from the moves because each change can save us one move of bringing a 1 closer
        return max(0, min_moves - min(maxChanges, k-1))