from typing import List, Set
from collections import defaultdict

class Solution:
    def shortestSubstrings(self, arr: List[str]) -> List[str]:
        # dictionary mapping each substring to the set of indices (from arr) where it appears
        substring_to_indices = defaultdict(set)
        
        # Precompute all distinct substrings for each string in arr and fill the dictionary
        n = len(arr)
        # We'll also store the distinct substrings for each word to avoid recomputation.
        substrings_per_word = [set() for _ in range(n)]
        
        for i, word in enumerate(arr):
            L = len(word)
            for start in range(L):
                for end in range(start+1, L+1):
                    substr = word[start:end]
                    # Only add if not already added for this word, as our stored set prevents duplicates
                    if substr not in substrings_per_word[i]:
                        substrings_per_word[i].add(substr)
                        substring_to_indices[substr].add(i)
        
        # Now, for each string in arr, we find the shortest substring that is unique to that string.
        answer = ["" for _ in range(n)]
        
        for i in range(n):
            candidates = list(substrings_per_word[i])
            # Sort candidates by length first then lexicographically (as required).
            candidates.sort(key=lambda s: (len(s), s))
            found = False
            for candidate in candidates:
                if substring_to_indices[candidate] == {i}:
                    answer[i] = candidate
                    found = True
                    break
            if not found:
                answer[i] = ""
        
        return answer