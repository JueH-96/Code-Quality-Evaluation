from typing import List
from collections import defaultdict

class Solution:
    def maxSubarrays(self, n: int, conflictingPairs: List[List[int]]) -> int:
        # Compute min_b array
        a_to_bs = defaultdict(list)
        for a, b in conflictingPairs:
            a_to_bs[a].append(b)
        
        min_b = [float('inf')] * (n + 2)  # min_b[0..n]
        for s in range(n, 0, -1):
            current_min = min_b[s + 1]
            if s in a_to_bs:
                current_min = min(current_min, min(a_to_bs[s]))
            min_b[s] = current_min
        
        # Compute original forbidden_area
        forbidden_area = 0
        for s in range(1, n + 1):
            if min_b[s] != float('inf'):
                forbidden_area += (n - min_b[s] + 1)
        
        original_valid = (n * (n + 1) // 2) - forbidden_area
        
        max_delta = 0
        for pair in conflictingPairs:
            a, b = pair
            area = a * (n - b + 1)
            overlap = 0
            for other in conflictingPairs:
                if other == pair:
                    continue
                c, d = other
                max_b = max(b, d)
                if max_b > n:
                    continue
                overlap += min(a, c) * (n - max_b + 1)
            delta = area - overlap
            if delta > max_delta:
                max_delta = delta
        
        return original_valid + max_delta