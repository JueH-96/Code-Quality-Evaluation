from typing import List

class Solution:
    def shortestSubstrings(self, arr: List[str]) -> List[str]:
        def get_substrings(s):
            substrings = set()
            for i in range(len(s)):
                for j in range(i + 1, len(s) + 1):
                    substrings.add(s[i:j])
            return substrings
        
        def find_shortest_unique_substring(s, other_substrings):
            substrings = get_substrings(s)
            unique_substrings = substrings - other_substrings
            if not unique_substrings:
                return ""
            unique_substrings = sorted(unique_substrings, key=lambda x: (len(x), x))
            return unique_substrings[0]
        
        n = len(arr)
        answer = [""] * n
        all_substrings = [get_substrings(s) for s in arr]
        
        for i in range(n):
            other_substrings = set()
            for j in range(n):
                if i != j:
                    other_substrings.update(all_substrings[j])
            answer[i] = find_shortest_unique_substring(arr[i], other_substrings)
        
        return answer