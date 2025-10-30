from math import gcd

class Solution:
    def findKthSmallest(self, coins: List[int], k: int) -> int:
        def lcm(a, b):
            return a * b // gcd(a, b)
        
        def count_multiples(x):
            # Count the number of distinct multiples <= x
            n = len(coins)
            count = 0
            
            # Iterate through all non-empty subsets
            for mask in range(1, 1 << n):
                # Find LCM of coins in this subset
                subset_lcm = 1
                for i in range(n):
                    if mask & (1 << i):
                        subset_lcm = lcm(subset_lcm, coins[i])
                        if subset_lcm > x:
                            break
                
                if subset_lcm <= x:
                    # Count of multiples of subset_lcm that are <= x
                    multiples = x // subset_lcm
                    
                    # Apply inclusion-exclusion principle
                    if bin(mask).count('1') % 2 == 1:
                        count += multiples
                    else:
                        count -= multiples
            
            return count
        
        # Binary search on the answer
        left, right = 1, k * min(coins)
        
        while left < right:
            mid = (left + right) // 2
            if count_multiples(mid) >= k:
                right = mid
            else:
                left = mid + 1
        
        return left