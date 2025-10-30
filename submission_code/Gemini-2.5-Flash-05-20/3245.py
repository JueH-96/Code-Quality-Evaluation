from typing import List

class Solution:
    def beautifulIndices(self, s: str, a: str, b: str, k: int) -> List[int]:
        
        # Helper function to find all starting indices of a pattern in a text.
        # This uses Python's built-in str.find which is highly optimized
        # (often implemented with algorithms like Boyer-Moore for efficiency).
        def find_all_occurrences(text: str, pattern: str) -> List[int]:
            occurrences = []
            start_index = 0
            while True:
                idx = text.find(pattern, start_index)
                if idx == -1:
                    break # No more occurrences found
                occurrences.append(idx)
                # To find all distinct starting indices, we must start the next search
                # from the character immediately after the current match's start.
                # For example, searching for "aa" in "aaa":
                # find("aa", 0) -> 0. Next search from index 1.
                # find("aa", 1) -> 1. Next search from index 2.
                # This ensures we get all valid starting positions, including overlapping ones.
                start_index = idx + 1 
            return occurrences

        # Step 1: Find all occurrences of string 'a' in 's'.
        # These indices will inherently be sorted as find_all_occurrences
        # processes the string from left to right.
        indices_a = find_all_occurrences(s, a)
        
        # Step 2: Find all occurrences of string 'b' in 's'.
        # These indices will also be inherently sorted.
        indices_b = find_all_occurrences(s, b)

        beautiful_indices = []
        
        # Step 3: For each index 'i' where 'a' is found, check if it's beautiful.
        # We use a two-pointer approach for 'indices_b' to efficiently check the |j - i| <= k condition.
        # This is efficient because both indices_a and indices_b are sorted.
        
        p_b = 0  # Pointer for indices_b

        for i in indices_a:
            # Move p_b forward to find the first index 'j' in indices_b such that j >= i - k.
            # We are looking for a 'j' in the range [i - k, i + k].
            # The 'while' loop advances p_b past all 'j's that are too far to the left of 'i'.
            while p_b < len(indices_b) and indices_b[p_b] < i - k:
                p_b += 1
            
            # After the above while loop, indices_b[p_b] (if p_b is still valid)
            # is the smallest 'j' such that j >= i - k.
            
            # Now, check if this 'j' is also within the upper bound (j <= i + k).
            # If p_b is within the bounds of indices_b AND indices_b[p_b] is not too far to the right,
            # then we have found at least one 'j' such that |j - i| <= k.
            # Thus, 'i' is a beautiful index.
            if p_b < len(indices_b) and indices_b[p_b] <= i + k:
                beautiful_indices.append(i)
                
        return beautiful_indices