class Solution:
    def minCostToEqualizeArray(self, nums: List[int], cost1: int, cost2: int) -> int:
        max_val = max(nums)
        n = len(nums)
        MOD = 10**9 + 7
        
        # If cost2 >= 2*cost1, it's always better to use operation 1
        if cost2 >= 2 * cost1:
            total_increments = sum(max(0, max_val - num) for num in nums)
            return (total_increments * cost1) % MOD
        
        min_cost = float('inf')
        target = max_val
        
        while target <= max_val + 2 * n:  # Safety cap
            # Calculate increments needed for each element
            diffs = [max(0, target - num) for num in nums]
            diffs = [d for d in diffs if d > 0]  # Keep only elements that need incrementing
            
            current_cost = 0
            
            # If only one element needs incrementing, we can only use operation 1
            if len(diffs) <= 1:
                current_cost = sum(diffs) * cost1
            else:
                # Sort diffs in descending order
                diffs.sort(reverse=True)
                
                # Process pairs of elements
                i = 0
                while i < len(diffs):
                    if i + 1 < len(diffs):
                        # Pair up elements i and i+1
                        min_val = min(diffs[i], diffs[i+1])
                        current_cost += min_val * cost2
                        
                        # Reduce both by min_val
                        diffs[i] -= min_val
                        diffs[i+1] -= min_val
                        
                        # Use operation 1 for any remaining increments
                        current_cost += diffs[i] * cost1
                        current_cost += diffs[i+1] * cost1
                    else:
                        # Last unpaired element
                        current_cost += diffs[i] * cost1
                    
                    i += 2
            
            if current_cost < min_cost:
                min_cost = current_cost
            else:
                break
            
            target += 1
        
        return min_cost % MOD