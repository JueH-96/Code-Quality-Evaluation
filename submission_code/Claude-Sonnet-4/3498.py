class Solution:
    def minChanges(self, nums: List[int], k: int) -> int:
        n = len(nums)
        pairs = []
        
        # Create pairs (i, n-1-i)
        for i in range(n // 2):
            pairs.append((nums[i], nums[n - 1 - i]))
        
        # Count frequency of each possible X value and track costs
        x_costs = {}  # x_value -> total_cost
        
        # For each possible X from 0 to k
        for x in range(k + 1):
            total_cost = 0
            
            for a, b in pairs:
                current_diff = abs(a - b)
                
                if current_diff == x:
                    # No changes needed
                    cost = 0
                else:
                    # Try changing one element
                    cost = 2  # Default to changing both
                    
                    # Try changing a to make |new_a - b| = x
                    new_a1 = b + x
                    new_a2 = b - x
                    if 0 <= new_a1 <= k or 0 <= new_a2 <= k:
                        cost = min(cost, 1)
                    
                    # Try changing b to make |a - new_b| = x
                    new_b1 = a + x
                    new_b2 = a - x
                    if 0 <= new_b1 <= k or 0 <= new_b2 <= k:
                        cost = min(cost, 1)
                
                total_cost += cost
            
            x_costs[x] = total_cost
        
        return min(x_costs.values())