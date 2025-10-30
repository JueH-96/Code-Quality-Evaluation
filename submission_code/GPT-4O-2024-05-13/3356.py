from typing import List

class Solution:
    def shortestSubstrings(self, arr: List[str]) -> List[str]:
        def get_all_substrings(s):
            substrings = set()
            for length in range(1, len(s) + 1):
                for start in range(len(s) - length + 1):
                    substrings.add(s[start:start + length])
            return substrings
        
        n = len(arr)
        all_substrings = [get_all_substrings(s) for s in arr]
        answer = []
        
        for i in range(n):
            unique_substrings = set()
            for substr in all_substrings[i]:
                is_unique = True
                for j in range(n):
                    if i != j and substr in all_substrings[j]:
                        is_unique = False
                        break
                if is_unique:
                    unique_substrings.add(substr)
            
            if unique_substrings:
                answer.append(min(unique_substrings, key=lambda x: (len(x), x)))
            else:
                answer.append("")
        
        return answer