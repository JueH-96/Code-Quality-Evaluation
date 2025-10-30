from typing import List

class Solution:
    def shortestSubstrings(self, arr: List[str]) -> List[str]:
        from collections import defaultdict
        
        def all_substrings(s):
            n = len(s)
            return {s[i:j] for i in range(n) for j in range(i + 1, n + 1)}
        
        substrings = defaultdict(set)
        for i, word in enumerate(arr):
            for substr in all_substrings(word):
                substrings[substr].add(i)
        
        answer = []
        for i, word in enumerate(arr):
            valid_substrings = {substr for substr in all_substrings(word) if len(substrings[substr]) == 1 and substrings[substr] == {i}}
            if valid_substrings:
                answer.append(min(valid_substrings))
            else:
                answer.append("")
        
        return answer