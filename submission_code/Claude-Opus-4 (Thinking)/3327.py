class Solution:
    def minimumMoves(self, nums: List[int], k: int, maxChanges: int) -> int:
        n = len(nums)
        min_moves = float('inf')
        
        for aliceIndex in range(n):
            # Free one at starting position
            free = 1 if nums[aliceIndex] == 1 else 0
            
            if free == k:
                return 0
            
            # Create list of options: (cost, is_created)
            options = []
            
            # Add all other positions as options
            for j in range(n):
                if j != aliceIndex:
                    if nums[j] == 1:
                        # Existing 1: cost is just the distance
                        options.append((abs(j - aliceIndex), False))
                    else:
                        # Can create 1 here: cost is 1 + distance
                        options.append((abs(j - aliceIndex) + 1, True))
            
            # Sort by cost to greedily pick cheapest options
            options.sort()
            
            # Greedy selection
            collected = free
            moves = 0
            used_changes = 0
            
            for cost, is_created in options:
                if collected == k:
                    break
                if is_created and used_changes >= maxChanges:
                    continue
                collected += 1
                moves += cost
                if is_created:
                    used_changes += 1
            
            if collected == k:
                min_moves = min(min_moves, moves)
        
        return min_moves