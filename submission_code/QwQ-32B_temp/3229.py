import bisect
from typing import List

class Solution:
    def minimumCost(self, nums: List[int]) -> int:
        nums_sorted = sorted(nums)
        n = len(nums_sorted)
        total_sum = sum(nums_sorted)
        prefix = [0] * (n + 1)
        for i in range(n):
            prefix[i + 1] = prefix[i] + nums_sorted[i]
        
        # Compute median
        median = nums_sorted[n // 2]
        
        # Collect unique elements
        unique_nums = list(set(nums))
        
        # Generate all candidates
        candidates = set()
        for num in unique_nums + [median]:
            pals = self.get_palindromes(num)
            for p in pals:
                candidates.add(p)
        
        # Now compute the minimal cost
        min_cost = float('inf')
        for y in candidates:
            pos = bisect.bisect_left(nums_sorted, y)
            cost = y * (2 * pos - n) + (total_sum - 2 * prefix[pos])
            if cost < min_cost:
                min_cost = cost
        
        return min_cost
    
    def get_palindromes(self, n: int) -> List[int]:
        s = str(n)
        length = len(s)
        candidates = []
        
        half = (length + 1) // 2
        for delta in (-1, 0, 1):
            prefix_str = str(int(s[:half]) + delta)
            if length % 2 == 0:
                pal = prefix_str + prefix_str[::-1]
            else:
                pal = prefix_str + prefix_str[:-1][::-1]
            candidates.append(int(pal))
        
        # Add edge cases
        edge1 = 10**(length - 1) - 1
        edge2 = 10**length + 1
        candidates.append(edge1)
        candidates.append(edge2)
        
        # Filter valid candidates (1 <= c < 1e9)
        valid = []
        for c in candidates:
            if 1 <= c < 10**9:
                valid.append(c)
        return valid