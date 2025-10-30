import math
import collections
import bisect
import heapq
import functools
import itertools
import sys
from typing import List 

# Optional: Set recursion depth if needed, though not for this problem.
# sys.setrecursionlimit(10**6) 

class Solution:
    """
    Finds beautiful indices in a string s based on occurrences of strings a and b
    within a distance k. An index i is beautiful if s starts with 'a' at i, and
    there exists an index j where s starts with 'b', such that |i - j| <= k.
    """
    def beautifulIndices(self, s: str, a: str, b: str, k: int) -> List[int]:
        """
        Args:
            s: The main string (0-indexed). Length up to 10^5.
            a: The first pattern string. Length up to 10.
            b: The second pattern string. Length up to 10.
            k: The maximum allowed absolute difference between indices i and j. 
               1 <= k <= s.length.

        Returns:
            A list containing all beautiful indices (indices 'i' that satisfy the condition) 
            in sorted order from smallest to largest.
        """
        
        n = len(s)
        la = len(a)
        lb = len(b)
        
        # Edge case: If patterns are longer than the string, they cannot possibly occur.
        # Although constraints state 1 <= length, this check is harmless.
        if la > n or lb > n:
            return []

        indices_a = []
        # Find all starting indices where pattern 'a' occurs in 's'.
        # We use str.find in a loop. The naive complexity is O(n * la), 
        # but since la <= 10, this is effectively O(n). Python's str.find 
        # might use more optimized algorithms internally (like Boyer-Moore).
        start_index = 0
        while True:
             # Find the next occurrence of 'a' starting from 'start_index'
             idx = s.find(a, start_index)
             if idx == -1: # No more occurrences found
                 break
             indices_a.append(idx)
             # Move start_index to the position right after the found index's start
             # to find the next distinct starting position.
             start_index = idx + 1 

        indices_b = []
        # Find all starting indices where pattern 'b' occurs in 's'.
        # Similar complexity analysis as for 'a', effectively O(n).
        start_index = 0
        while True:
             # Find the next occurrence of 'b' starting from 'start_index'
             idx = s.find(b, start_index)
             if idx == -1: # No more occurrences found
                 break
             indices_b.append(idx)
             # Move start_index forward.
             start_index = idx + 1

        # If either pattern 'a' or 'b' does not occur in 's', then no index 'i' 
        # can satisfy the condition that requires both 'a' at 'i' and 'b' at some 'j'.
        if not indices_a or not indices_b:
            return []
            
        result = [] # This list will store the beautiful indices found.
        ptr_b = 0 # Initialize a pointer for scanning the indices_b list.
        n_b = len(indices_b) # Cache the length of indices_b for efficiency.
        
        # Iterate through each index 'i' where pattern 'a' starts. These indices are sorted.
        # Use a two-pointer approach (i iterates through indices_a, ptr_b scans indices_b)
        # to efficiently check the distance constraint |j - i| <= k.
        # The complexity of this part is O(len(indices_a) + len(indices_b)), which is O(n).
        for i in indices_a:
            # Advance the pointer 'ptr_b' forward in indices_b until we find the first index 'j'
            # such that j >= i - k. This means we are looking for a 'j' that is not too small.
            # Since 'i' is increasing in the outer loop, 'ptr_b' only needs to move forward.
            while ptr_b < n_b and indices_b[ptr_b] < i - k:
                ptr_b += 1
                
            # After the while loop, if ptr_b is still within bounds (ptr_b < n_b), 
            # indices_b[ptr_b] is the smallest 'j' that satisfies j >= i - k.
            
            # Now, we check if this 'j' also satisfies the other half of the condition: j <= i + k.
            # We must also ensure that 'ptr_b' did not go out of bounds.
            if ptr_b < n_b and indices_b[ptr_b] <= i + k:
                # If both conditions (j >= i - k and j <= i + k) are met by indices_b[ptr_b],
                # it means we have found a suitable index 'j' for the current index 'i'.
                # Therefore, 'i' is a beautiful index. Add it to the result list.
                result.append(i)
                
        # The 'result' list contains all beautiful indices 'i'. Since we iterated through 
        # 'indices_a' (which is sorted) and added 'i' values in that order, the 'result' 
        # list is already sorted in ascending order.
        return result