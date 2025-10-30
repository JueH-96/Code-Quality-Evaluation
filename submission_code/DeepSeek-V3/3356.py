from typing import List

class Solution:
    def shortestSubstrings(self, arr: List[str]) -> List[str]:
        n = len(arr)
        answer = [""] * n
        
        # Precompute all possible substrings for each string
        all_substrings = []
        for s in arr:
            substrings = set()
            length = len(s)
            for l in range(1, length + 1):
                for i in range(length - l + 1):
                    substrings.add(s[i:i+l])
            all_substrings.append(substrings)
        
        for i in range(n):
            s = arr[i]
            length = len(s)
            # Generate all possible substrings of s, ordered by length and lex order
            substrings = []
            for l in range(1, length + 1):
                for j in range(length - l + 1):
                    sub = s[j:j+l]
                    substrings.append((l, sub))
            # Sort by length, then lex order
            substrings.sort()
            # Check each substring in order
            for l, sub in substrings:
                # Check if this substring is not in any other string's substrings
                is_unique = True
                for k in range(n):
                    if k != i and sub in all_substrings[k]:
                        is_unique = False
                        break
                if is_unique:
                    answer[i] = sub
                    break
        return answer