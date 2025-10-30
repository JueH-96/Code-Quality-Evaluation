class Solution:
    def findKthSmallest(self, coins: List[int], k: int) -> int:
        from math import gcd
        from functools import reduce
        
        def lcm(a, b):
            return a * b // gcd(a, b)
        
        def count_multiples(x):
            # Count how many numbers <= x are divisible by at least one coin
            n = len(coins)
            count = 0
            
            # Use bit manipulation to generate all subsets
            for mask in range(1, 1 << n):
                subset_lcm = 1
                subset_size = 0
                
                for i in range(n):
                    if mask & (1 << i):
                        subset_lcm = lcm(subset_lcm, coins[i])
                        subset_size += 1
                        # Optimization: if LCM exceeds x, no need to continue
                        if subset_lcm > x:
                            break
                
                if subset_lcm <= x:
                    # Inclusion-exclusion: add if odd size, subtract if even
                    if subset_size % 2 == 1:
                        count += x // subset_lcm
                    else:
                        count -= x // subset_lcm
            
            return count
        
        # Binary search for the k-th smallest
        left, right = 1, k * min(coins)
        
        while left < right:
            mid = (left + right) // 2
            if count_multiples(mid) < k:
                left = mid + 1
            else:
                right = mid
        
        return left