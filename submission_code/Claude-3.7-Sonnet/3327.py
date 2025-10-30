class Solution:
    def minimumMoves(self, nums: List[int], k: int, maxChanges: int) -> int:
        n = len(nums)
        
        # Edge case
        if k == 0:
            return 0
        
        min_cost = float('inf')
        
        for aliceIndex in range(n):
            cost = 0
            remaining_k = k
            
            # 1. If aliceIndex has a one, pick it up immediately (0 moves)
            if nums[aliceIndex] == 1:
                remaining_k -= 1
            
            if remaining_k == 0:
                return 0
            
            # 2. Pick up adjacent ones (1 move each)
            adj_ones = 0
            if aliceIndex > 0 and nums[aliceIndex - 1] == 1:
                adj_ones += 1
            if aliceIndex < n - 1 and nums[aliceIndex + 1] == 1:
                adj_ones += 1
            
            picked_adj_ones = min(adj_ones, remaining_k)
            cost += picked_adj_ones
            remaining_k -= picked_adj_ones
            
            if remaining_k == 0:
                min_cost = min(min_cost, cost)
                continue
            
            # 3. Use action 1 + action 2 for the rest (2 moves each)
            action1_used = min(remaining_k, maxChanges)
            cost += 2 * action1_used
            remaining_k -= action1_used
            
            if remaining_k == 0:
                min_cost = min(min_cost, cost)
                continue
            
            # 4. Pick up remaining ones from further away
            distances = []
            for i in range(n):
                if nums[i] == 1 and i != aliceIndex and abs(i - aliceIndex) > 1:
                    distances.append(abs(i - aliceIndex))
            
            distances.sort()
            if len(distances) < remaining_k:
                continue  # Can't pick up enough ones
            
            for i in range(remaining_k):
                cost += distances[i]
            
            min_cost = min(min_cost, cost)
        
        return min_cost