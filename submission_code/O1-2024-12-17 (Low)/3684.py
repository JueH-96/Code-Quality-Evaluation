class Solution:
    def hasMatch(self, s: str, p: str) -> bool:
        # Find the position of '*' in p
        star_index = p.index('*')
        # Split p into prefix (before '*') and suffix (after '*')
        prefix = p[:star_index]
        suffix = p[star_index+1:]

        # The minimum length of a substring in s that can fit prefix + suffix
        min_len = len(prefix) + len(suffix)

        # Check all possible substrings of s
        n = len(s)
        for start in range(n):
            for end in range(start + min_len, n + 1):
                # current substring from s[start:end]
                sub = s[start:end]
                # Check if the current substring can match the pattern
                # Match prefix at the start and suffix at the end
                if sub.startswith(prefix) and sub.endswith(suffix):
                    return True

        return False