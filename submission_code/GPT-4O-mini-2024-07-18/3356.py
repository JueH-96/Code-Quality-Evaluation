from typing import List

class Solution:
    def shortestSubstrings(self, arr: List[str]) -> List[str]:
        def get_all_substrings(s):
            substrings = set()
            n = len(s)
            for length in range(1, n + 1):
                for start in range(n - length + 1):
                    substrings.add(s[start:start + length])
            return substrings
        
        # Get all substrings from all strings in arr
        all_substrings = set()
        for s in arr:
            all_substrings.update(get_all_substrings(s))
        
        answer = []
        
        for s in arr:
            found = None
            # Check for the shortest unique substring
            for length in range(1, len(s) + 1):
                for start in range(len(s) - length + 1):
                    substring = s[start:start + length]
                    if substring not in all_substrings or substring == s:
                        if found is None or (len(substring) < len(found)) or (len(substring) == len(found) and substring < found):
                            found = substring
            answer.append(found if found is not None else "")
        
        return answer