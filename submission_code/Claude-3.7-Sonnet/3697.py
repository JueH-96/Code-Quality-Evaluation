class Solution:
    def minimumIncrements(self, nums: List[int], target: List[int]) -> int:
        from itertools import product
        from math import gcd
        
        def lcm(a, b):
            return a * b // gcd(a, b)
        
        # Try all possible assignments of target elements to nums elements
        min_total_increments = float('inf')
        for assignment in product(range(len(nums)), repeat=len(target)):
            target_assignment = {}
            for t_idx, n_idx in enumerate(assignment):
                if n_idx not in target_assignment:
                    target_assignment[n_idx] = []
                target_assignment[n_idx].append(target[t_idx])
            
            total_increments = 0
            for n_idx, targets_assigned in target_assignment.items():
                n = nums[n_idx]
                
                # Find the LCM of all targets assigned to this nums element
                current_lcm = targets_assigned[0]
                for t in targets_assigned[1:]:
                    current_lcm = lcm(current_lcm, t)
                
                # Calculate the increments needed to make n a multiple of the LCM
                increments = (current_lcm - (n % current_lcm)) % current_lcm
                total_increments += increments
            
            min_total_increments = min(min_total_increments, total_increments)
        
        return min_total_increments