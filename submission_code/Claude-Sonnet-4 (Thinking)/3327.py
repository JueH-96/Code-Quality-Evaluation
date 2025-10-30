class Solution:
    def minimumMoves(self, nums: List[int], k: int, maxChanges: int) -> int:
        n = len(nums)
        ones = [i for i in range(n) if nums[i] == 1]
        
        min_moves = float('inf')
        
        for alice in range(n):
            needed = k
            if nums[alice] == 1:
                needed -= 1
            
            if needed == 0:
                return 0
            
            # Calculate distances to all other ones
            distances = []
            for pos in ones:
                if pos != alice:
                    distances.append(abs(pos - alice))
            distances.sort()
            
            # Precompute prefix sums for efficient range sum queries
            prefix_sum = [0]
            for d in distances:
                prefix_sum.append(prefix_sum[-1] + d)
            
            # Try all possible numbers of creations (0 to min(needed, maxChanges))
            for create_count in range(min(needed, maxChanges) + 1):
                collect_count = needed - create_count
                
                if collect_count > len(distances):
                    continue  # Not enough existing ones to collect
                
                cost = 2 * create_count + prefix_sum[collect_count]
                min_moves = min(min_moves, cost)
        
        return min_moves