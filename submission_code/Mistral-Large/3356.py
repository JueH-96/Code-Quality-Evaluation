from typing import List

class Solution:
    def shortestSubstrings(self, arr: List[str]) -> List[str]:
        def is_unique_substring(substring, arr, index):
            for i in range(len(arr)):
                if i != index and substring in arr[i]:
                    return False
            return True

        def lexicographically_smallest_substring(s, arr, index):
            substrings = sorted((s[i:j] for i in range(len(s)) for j in range(i + 1, len(s) + 1)), key=lambda x: (len(x), x))
            for substring in substrings:
                if is_unique_substring(substring, arr, index):
                    return substring
            return ""

        result = []
        for i in range(len(arr)):
            result.append(lexicographically_smallest_substring(arr[i], arr, i))

        return result