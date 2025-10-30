from typing import List
import math

class Solution:
    def assignElements(self, groups: List[int], elements: List[int]) -> List[int]:
        max_val = 100000  # given constraints: groups[i], elements[i] up to 100000

        # Precompute the smallest index for each element value present in elements.
        # For each value v (1 to max_val) we store the minimum index j such that elements[j] == v.
        min_index = [None] * (max_val + 1)
        for j, v in enumerate(elements):
            if min_index[v] is None or j < min_index[v]:
                min_index[v] = j

        # For each number from 1 to max_val, we want to know the candidate (minimal index j)
        # among all elements that can be assigned if a group value is divisible by that element.
        # We'll precompute a list candidate[m] for every possible m in [1, max_val]
        candidate = [float('inf')] * (max_val + 1)
        
        # For every potential divisor d that appears in elements, update all multiples m of d.
        for d in range(1, max_val + 1):
            if min_index[d] is not None:
                # iterate over multiples of d: m = d, 2*d, 3*d, ... up to max_val.
                for m in range(d, max_val + 1, d):
                    candidate[m] = min(candidate[m], min_index[d])
        
        # Now, for each group in groups, candidate[group] will be the minimum index of an element
        # in elements such that group % element == 0. If no such element exists, candidate[group] will be float('inf').
        result = []
        for g in groups:
            if candidate[g] == float('inf'):
                result.append(-1)
            else:
                result.append(candidate[g])
        return result