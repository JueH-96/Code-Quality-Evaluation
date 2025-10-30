class Solution:
    def findKthSmallest(self, coins: List[int], k: int) -> int:
        def gcd(a, b):
            while b:
                a, b = b, a % b
            return a
        
        def lcm(a, b):
            return a * b // gcd(a, b)
        
        def lcm_of_list(lst):
            result = lst[0]
            for i in range(1, len(lst)):
                result = lcm(result, lst[i])
            return result
        
        def count_values_le(mid):
            n = len(coins)
            total = 0
            
            # Inclusion-exclusion principle
            for mask in range(1, 1 << n):
                subset = []
                for i in range(n):
                    if mask & (1 << i):
                        subset.append(coins[i])
                
                lcm_subset = lcm_of_list(subset)
                count = mid // lcm_subset
                
                if len(subset) % 2 == 1:
                    total += count
                else:
                    total -= count
            
            return total
        
        left, right = 1, k * min(coins)
        while left < right:
            mid = (left + right) // 2
            if count_values_le(mid) >= k:
                right = mid
            else:
                left = mid + 1
        
        return left