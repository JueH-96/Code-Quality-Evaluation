from typing import List

class Solution:
    def shortestSubstrings(self, arr: List[str]) -> List[str]:
        def is_substring(s: str, arr: List[str]) -> bool:
            """Check if a string is a substring of any string in the array."""
            for string in arr:
                if s != string and s in string:
                    return True
            return False

        def get_shortest_substring(s: str, arr: List[str]) -> str:
            """Get the shortest substring that does not occur in any other string."""
            substrings = []
            for length in range(1, len(s) + 1):
                for i in range(len(s) - length + 1):
                    substring = s[i:i + length]
                    if not is_substring(substring, arr):
                        substrings.append(substring)
            return min(substrings, default='') if substrings else ''

        return [get_shortest_substring(s, arr) for s in arr]