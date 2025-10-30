from typing import List

class Solution:
    def shortestSubstrings(self, arr: List[str]) -> List[str]:
        # Build a mapping from substring to the set of string indices in which it appears.
        substr_occurrences = {}
        for i, s in enumerate(arr):
            seen = set()  # To avoid processing duplicate substrings in the same string
            for start in range(len(s)):
                for end in range(start + 1, len(s) + 1):
                    sub = s[start:end]
                    if sub in seen:
                        continue
                    seen.add(sub)
                    if sub not in substr_occurrences:
                        substr_occurrences[sub] = set()
                    substr_occurrences[sub].add(i)
        
        n = len(arr)
        answer = ["" for _ in range(n)]
        
        # For each string, check for the shortest substring (and lexicographically smallest among equals)
        # that doesn't occur in any other string.
        for i, s in enumerate(arr):
            found = False
            # Try candidate substrings in increasing length order.
            for curr_len in range(1, len(s) + 1):
                candidates = set()
                for start in range(len(s) - curr_len + 1):
                    candidates.add(s[start:start + curr_len])
                for candidate in sorted(candidates):
                    # A candidate is valid if it appears only in the current string.
                    if substr_occurrences.get(candidate, set()) == {i}:
                        answer[i] = candidate
                        found = True
                        break
                if found:
                    break
        
        return answer