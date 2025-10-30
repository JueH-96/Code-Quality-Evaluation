from typing import List

class Solution:
    def shortestSubstrings(self, arr: List[str]) -> List[str]:
        # Count in how many distinct strings each substring appears.
        substr_count = {}
        for s in arr:
            seen = set()
            L = len(s)
            # Generate all substrings of s, but count each at most once per string
            for i in range(L):
                for j in range(i+1, L+1):
                    sub = s[i:j]
                    if sub not in seen:
                        seen.add(sub)
                        substr_count[sub] = substr_count.get(sub, 0) + 1
        
        result = []
        # For each string, find the shortest unique substring
        for s in arr:
            L = len(s)
            answer = ""
            found = False
            # Try substrings in increasing length
            for length in range(1, L+1):
                seen = set()
                candidates = []
                # Slide window of size `length`
                for i in range(L - length + 1):
                    sub = s[i:i+length]
                    if sub in seen:
                        continue
                    seen.add(sub)
                    # If this substring appears only in this one string:
                    if substr_count.get(sub, 0) == 1:
                        candidates.append(sub)
                if candidates:
                    # Pick lexicographically smallest among the shortest
                    answer = min(candidates)
                    found = True
                    break
            result.append(answer if found else "")
        
        return result