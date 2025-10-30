class Solution:
    def minimumMoves(self, nums: List[int], k: int, maxChanges: int) -> int:
        # Find positions of all 1s
        ones = [i for i, x in enumerate(nums) if x == 1]
        
        if not ones:
            # No existing 1s, must create all k ones
            return k * 2
        
        min_moves = float('inf')
        
        # Try each position as Alice's starting position
        for alice_pos in range(len(nums)):
            # Calculate cost for this starting position
            moves = 0
            collected = 0
            
            # If starting position has a 1, collect it for free
            if nums[alice_pos] == 1:
                collected += 1
            
            if collected >= k:
                return 0
            
            remaining = k - collected
            
            # Calculate distances to all other 1s
            distances = []
            for pos in ones:
                if pos != alice_pos:
                    distances.append(abs(pos - alice_pos))
            
            distances.sort()
            
            # Try different strategies: use x existing 1s and create (remaining-x) new ones
            for use_existing in range(min(len(distances), remaining) + 1):
                create_new = remaining - use_existing
                
                if create_new <= maxChanges:
                    cost = sum(distances[:use_existing]) + create_new * 2
                    min_moves = min(min_moves, cost)
        
        return min_moves