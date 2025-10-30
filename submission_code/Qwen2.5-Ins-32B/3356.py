from typing import List

class Solution:
    def shortestSubstrings(self, arr: List[str]) -> List[str]:
        def all_substrings(s):
            return {s[i:j] for i in range(len(s)) for j in range(i + 1, len(s) + 1)}
        
        def find_shortest_unique_substring(s, others):
            substrings = all_substrings(s)
            for sub in sorted(substrings, key=lambda x: (len(x), x)):
                if all(sub not in other for other in others):
                    return sub
            return ""
        
        result = []
        for i, s in enumerate(arr):
            others = arr[:i] + arr[i+1:]
            result.append(find_shortest_unique_substring(s, others))
        return result