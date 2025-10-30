from typing import List
from collections import Counter

class Solution:
    def countPairs(self, coordinates: List[List[int]], k: int) -> int:
        # Build frequency dictionary for coordinates as tuples.
        freq = Counter((x, y) for x, y in coordinates)
        
        # We'll count pairs in a way that each unordered pair is counted twice (if two distinct keys)
        # so at the end we will divide by 2.
        total = 0
        
        # For each unique coordinate (x, y) and each possible a in [0, k] (where b = k - a)
        # candidate coordinate (tx, ty) is defined as (x XOR a, y XOR b)
        # Then the XOR distance between (x,y) and (tx,ty) becomes:
        #    (x XOR (x XOR a)) + (y XOR (y XOR b)) = a + b = k.
        for (x, y), count1 in freq.items():
            for a in range(k + 1):
                b = k - a
                tx = x ^ a
                ty = y ^ b
                candidate = (tx, ty)
                if candidate in freq:
                    count2 = freq[candidate]
                    # if candidate is the same coordinate as (x,y) then (x XOR 0, y XOR k) must equal (x,y)
                    # Notice: x ^ a equals x only if a == 0. Hence candidate can equal (x,y) only in the case a==0.
                    if candidate == (x, y):
                        # count pairs within same group (combination count)
                        # Each pair (i,j) is counted exactly once in this branch.
                        total += count1 * (count1 - 1)
                    else:
                        total += count1 * count2
        # In the above loop:
        # - For candidate different from (x,y), each unordered pair is counted twice:
        #   once from key and once from candidate.
        # - For candidate equal to (x,y), we added n*(n-1) which counts each pair exactly twice 
        #   (ordered pairs i,j and j,i) so we want to divide overall sum by 2.
        return total // 2