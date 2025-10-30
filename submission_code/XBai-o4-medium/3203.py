from typing import List
from collections import Counter

class Solution:
    def canMakePalindromeQueries(self, s: str, queries: List[List[int]]) -> List[bool]:
        n = len(s)
        mid = n // 2
        
        # Check if left and right halves have the same character counts
        left_str = s[:mid]
        right_str = s[mid:]
        if Counter(left_str) != Counter(right_str):
            return [False] * len(queries)
        
        # Precompute fixed_pairs array
        fixed_pairs = [0] * mid
        for i in range(mid):
            mirror_i = n - 1 - i
            if s[i] != s[mirror_i]:
                fixed_pairs[i] = 1
        
        # Compute prefix sum array
        prefix = [0] * (mid + 1)
        for i in range(mid):
            prefix[i + 1] = prefix[i] + fixed_pairs[i]
        
        total_fixed_false = prefix[mid]
        
        # If all pairs are already mirrored, return all True
        if total_fixed_false == 0:
            return [True] * len(queries)
        
        # Process each query
        result = []
        for a, b, c, d in queries:
            # Compute x and y for the mirrored interval of [c, d]
            x = (n - 1) - d
            y = (n - 1) - c
            
            # Function to compute range sum
            def range_sum(l, r):
                if l > r:
                    return 0
                return prefix[r + 1] - prefix[l]
            
            count1 = range_sum(a, b)
            count2 = range_sum(x, y)
            
            # Compute intersection between [a, b] and [x, y]
            start_overlap = max(a, x)
            end_overlap = min(b, y)
            if start_overlap > end_overlap:
                count_inter = 0
            else:
                count_inter = range_sum(start_overlap, end_overlap)
            
            union_count = count1 + count2 - count_inter
            
            if (total_fixed_false - union_count) == 0:
                result.append(True)
            else:
                result.append(False)
        
        return result