import collections
from typing import List

class Solution:
    def shortestSubstrings(self, arr: List[str]) -> List[str]:
        n = len(arr)
        
        # Step 1: Preprocessing - Build occurrences map
        # occurrences[substring] = set of indices where this substring appears
        # e.g., occurrences["ab"] = {0, 2} means "ab" is a substring of arr[0] and arr[2]
        occurrences = collections.defaultdict(set)
        
        # Iterate through each string and its substrings to populate the map
        for i, s in enumerate(arr):
            # Generate all possible substrings for the current string s
            for length in range(1, len(s) + 1): # Substring length from 1 to len(s)
                for j in range(len(s) - length + 1): # Starting index for substring
                    sub = s[j : j + length]
                    occurrences[sub].add(i)
        
        # Step 2: Find the shortest unique substring for each arr[i]
        answer = [""] * n # Initialize answer array with empty strings
        
        for i, s in enumerate(arr):
            # Generate all substrings for the current string arr[i]
            # We need to collect them first to sort them before checking for uniqueness.
            all_substrings_of_s = []
            for length in range(1, len(s) + 1):
                for j in range(len(s) - length + 1):
                    sub = s[j : j + length]
                    all_substrings_of_s.append(sub)
            
            # Sort the substrings. The primary sort key is length (shortest first),
            # and the secondary sort key is lexicographical order (smallest first).
            # This ensures that when we iterate, the first unique substring found
            # will be the one meeting all criteria (shortest, then lexicographically smallest).
            all_substrings_of_s.sort(key=lambda x: (len(x), x))
            
            # Iterate through the sorted substrings
            for current_sub in all_substrings_of_s:
                # A substring is "unique" for arr[i] if it only occurs in arr[i].
                # This is true if the set of its occurrences contains only 'i' and has a size of 1.
                # The condition 'i in occurrences[current_sub]' is implicitly true if len is 1,
                # as current_sub was derived from arr[i].
                if len(occurrences[current_sub]) == 1:
                    answer[i] = current_sub
                    # Found the best candidate for arr[i], so no need to check further substrings
                    # for this arr[i]. Move to the next string in the input array.
                    break 
            
        return answer