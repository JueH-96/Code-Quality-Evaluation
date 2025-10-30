class Solution:
    def findKthSmallest(self, coins: List[int], k: int) -> int:
        from math import gcd
        from functools import reduce
        
        def lcm(a, b):
            return a * b // gcd(a, b)
        
        def lcm_multiple(nums):
            return reduce(lcm, nums)
        
        def count_amounts_up_to(target):
            # Count how many amounts <= target can be made
            # Use inclusion-exclusion principle
            n = len(coins)
            total = 0
            
            # Iterate through all non-empty subsets
            for mask in range(1, 1 << n):
                subset = []
                for i in range(n):
                    if mask & (1 << i):
                        subset.append(coins[i])
                
                # Calculate LCM of current subset
                current_lcm = lcm_multiple(subset)
                
                # Add or subtract based on inclusion-exclusion
                # Odd size subset: add, even size subset: subtract
                if len(subset) % 2 == 1:
                    total += target // current_lcm
                else:
                    total -= target // current_lcm
            
            return total
        
        # Binary search for the k-th smallest amount
        left, right = 1, k * min(coins)
        
        while left < right:
            mid = (left + right) // 2
            if count_amounts_up_to(mid) < k:
                left = mid + 1
            else:
                right = mid
        
        return left