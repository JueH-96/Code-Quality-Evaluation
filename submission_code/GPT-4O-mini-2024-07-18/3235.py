from typing import List
from collections import defaultdict
import sys

class Solution:
    def minimumCost(self, source: str, target: str, original: List[str], changed: List[str], cost: List[int]) -> int:
        n = len(source)
        if n != len(target):
            return -1
        
        # Create a mapping from original characters to their possible changes and costs
        change_map = defaultdict(list)
        for o, c, co in zip(original, changed, cost):
            change_map[o].append((c, co))
        
        total_cost = 0
        
        # Dictionary to keep track of the minimum cost to convert each character
        min_cost = {}
        
        for i in range(n):
            s_char = source[i]
            t_char = target[i]
            
            if s_char == t_char:
                continue
            
            if s_char not in change_map:
                return -1
            
            # We need to find the minimum cost to convert s_char to t_char
            min_conversion_cost = sys.maxsize
            
            # Check all possible conversions from s_char
            for (c, co) in change_map[s_char]:
                if c == t_char:
                    min_conversion_cost = min(min_conversion_cost, co)
                else:
                    # If we can convert to c, check if we can convert c to t_char
                    if c in change_map:
                        for (next_c, next_co) in change_map[c]:
                            if next_c == t_char:
                                min_conversion_cost = min(min_conversion_cost, co + next_co)
            
            if min_conversion_cost == sys.maxsize:
                return -1
            
            total_cost += min_conversion_cost
        
        return total_cost