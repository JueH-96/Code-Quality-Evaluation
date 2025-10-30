class Solution:
    def minimumMoves(self, nums: List[int], k: int, maxChanges: int) -> int:
        n = len(nums)
        
        # Get positions of all 1s
        ones = [i for i in range(n) if nums[i] == 1]
        
        if not ones and k > 0:
            # All zeros, must use maxChanges
            return 2 * k
        
        min_moves = float('inf')
        
        # Try each possible starting position for Alice
        for alice_pos in range(n):
            moves = 0
            collected = 0
            
            # Count free pickups (at position and adjacent)
            free_pickups = 0
            if alice_pos < n and nums[alice_pos] == 1:
                free_pickups += 1
            if alice_pos > 0 and nums[alice_pos - 1] == 1:
                free_pickups += 1
            if alice_pos < n - 1 and nums[alice_pos + 1] == 1:
                free_pickups += 1
            
            # Collect free ones
            collected = min(free_pickups, k)
            
            if collected >= k:
                min_moves = 0
                continue
            
            # Use maxChanges (each costs 2 moves)
            changes_used = min(maxChanges, k - collected)
            collected += changes_used
            moves += 2 * changes_used
            
            if collected >= k:
                min_moves = min(min_moves, moves)
                continue
            
            # Need to move existing 1s
            remaining = k - collected
            
            # Calculate distances to all 1s (excluding already collected)
            distances = []
            for pos in ones:
                if pos == alice_pos:
                    continue
                if abs(pos - alice_pos) == 1:
                    continue
                distances.append(abs(pos - alice_pos))
            
            distances.sort()
            
            # Take the closest ones
            for i in range(min(remaining, len(distances))):
                moves += distances[i]
            
            if collected + min(remaining, len(distances)) >= k:
                min_moves = min(min_moves, moves)
        
        return min_moves