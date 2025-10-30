class Solution:
    def minimumSubstringsInPartition(self, s: str) -> int:
        def is_balanced(substring: str) -> bool:
            """Check if a substring is balanced."""
            char_count = {}
            for char in substring:
                if char in char_count:
                    char_count[char] += 1
                else:
                    char_count[char] = 1
            return len(set(char_count.values())) == 1

        def dfs(start: int, path: list) -> int:
            """Perform depth-first search to find the minimum number of substrings."""
            if start == len(s):
                return len(path)
            min_substrings = float('inf')
            for end in range(start + 1, len(s) + 1):
                substring = s[start:end]
                if is_balanced(substring):
                    min_substrings = min(min_substrings, dfs(end, path + [substring]))
            return min_substrings

        return dfs(0, [])