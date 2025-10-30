from collections import defaultdict
from typing import List

class Solution:
    def shortestSubstrings(self, arr: List[str]) -> List[str]:
        substr_indices = defaultdict(set)
        n = len(arr)
        
        # Preprocess all substrings and track their indices
        for i in range(n):
            s = arr[i]
            for start in range(len(s)):
                for end in range(start + 1, len(s) + 1):
                    substr = s[start:end]
                    substr_indices[substr].add(i)
        
        answer = []
        for i in range(n):
            current_str = arr[i]
            candidates = []
            # Generate all possible substrings for the current string
            for start in range(len(current_str)):
                for end in range(start + 1, len(current_str) + 1):
                    substr = current_str[start:end]
                    candidates.append(substr)
            # Sort substrings by length and lexicographical order
            candidates.sort(key=lambda x: (len(x), x))
            found = False
            for substr in candidates:
                # Check if the substring is unique to the current string
                if substr_indices[substr] == {i}:
                    answer.append(substr)
                    found = True
                    break
            if not found:
                answer.append("")
        return answer