from typing import List

class Solution:
    def assignElements(self, groups: List[int], elements: List[int]) -> List[int]:
        val_to_min_idx = {}
        for j, val in enumerate(elements):
            if val not in val_to_min_idx:
                val_to_min_idx[val] = j
        
        result = []
        for g in groups:
            min_idx = float('inf')
            divisors = set()
            # Generate all divisors of g
            for i in range(1, int(g**0.5) + 1):
                if g % i == 0:
                    divisors.add(i)
                    divisors.add(g // i)
            # Check each divisor in the map
            for d in divisors:
                if d in val_to_min_idx:
                    if val_to_min_idx[d] < min_idx:
                        min_idx = val_to_min_idx[d]
            if min_idx != float('inf'):
                result.append(min_idx)
            else:
                result.append(-1)
        return result