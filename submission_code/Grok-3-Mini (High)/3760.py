import math
from typing import List

class Solution:
    def assignElements(self, groups: List[int], elements: List[int]) -> List[int]:
        # Build a dictionary to store the smallest index for each unique element value
        min_index = {}
        for idx, val in enumerate(elements):
            if val not in min_index:
                min_index[val] = idx
        
        # Define a function to get all divisors of a number
        def get_divisors(num):
            divs = []
            for i in range(1, int(math.sqrt(num)) + 1):
                if num % i == 0:
                    divs.append(i)
                    if i != num // i:
                        divs.append(num // i)
            return divs
        
        # Result list to store the assigned indices
        res = []
        
        # For each group, find the smallest index element that divides the group size
        for g in groups:
            divs = get_divisors(g)
            min_idx_found = float('inf')
            for div in divs:
                if div in min_index:
                    idx = min_index[div]
                    if idx < min_idx_found:
                        min_idx_found = idx
            if min_idx_found == float('inf'):
                res.append(-1)
            else:
                res.append(min_idx_found)
        
        return res