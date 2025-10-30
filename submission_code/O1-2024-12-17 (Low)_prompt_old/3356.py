from typing import List

class Solution:
    def shortestSubstrings(self, arr: List[str]) -> List[str]:
        n = len(arr)

        # Precompute all substrings for each string (including duplicates),
        # but store them in a way that we can iterate in both length and lex order.
        # For each string s, we'll collect its substrings in a sorted list
        # (sorted first by length, then lexicographically).
        all_substrings = []
        for s in arr:
            subs = []
            length = len(s)
            for start in range(length):
                for end in range(start + 1, length + 1):
                    subs.append(s[start:end])
            # Sort by (length, lexicographical order)
            subs.sort(key=lambda x: (len(x), x))
            all_substrings.append(subs)

        # For each string s, we will find the first substring in the sorted list
        # that does not appear in any other string.
        result = []
        for i in range(n):
            unique_substring = ""
            s_subs = all_substrings[i]
            found = False
            for sub in s_subs:
                # Check if sub appears in any other string arr[j], j != i
                is_unique = True
                for j in range(n):
                    if j == i:
                        continue
                    if sub in arr[j]:
                        is_unique = False
                        break
                if is_unique:
                    unique_substring = sub
                    found = True
                    break
            if not found:
                unique_substring = ""
            result.append(unique_substring)

        return result