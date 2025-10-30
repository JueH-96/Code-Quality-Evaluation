import itertools

class Solution:
    def minimumString(self, a: str, b: str, c: str) -> str:
        from itertools import permutations

        def merge(s1, s2):
            # Find the maximum overlap where s1 suffix matches s2 prefix
            max_overlap = 0
            min_len = min(len(s1), len(s2))
            for i in range(1, min_len + 1):
                if s1[-i:] == s2[:i]:
                    max_overlap = i
            return s1 + s2[max_overlap:]

        # Remove strings that are substrings of others
        strings = [a, b, c]
        unique_strings = []
        for s in strings:
            if not any(s != t and s in t for t in strings):
                unique_strings.append(s)

        # Now, consider all permutations of the unique_strings
        min_str = None
        for perm in permutations(unique_strings):
            merged = perm[0]
            for i in range(1, len(perm)):
                merged = merge(merged, perm[i])
            if min_str is None or len(merged) < len(min_str) or (len(merged) == len(min_str) and merged < min_str):
                min_str = merged

        return min_str