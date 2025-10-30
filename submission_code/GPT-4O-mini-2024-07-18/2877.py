class Solution:
    def minimumString(self, a: str, b: str, c: str) -> str:
        def merge_strings(s1, s2):
            max_overlap = 0
            for i in range(1, min(len(s1), len(s2)) + 1):
                if s1[-i:] == s2[:i]:
                    max_overlap = i
            return s1 + s2[max_overlap:]

        candidates = [
            merge_strings(merge_strings(a, b), c),
            merge_strings(merge_strings(a, c), b),
            merge_strings(merge_strings(b, a), c),
            merge_strings(merge_strings(b, c), a),
            merge_strings(merge_strings(c, a), b),
            merge_strings(merge_strings(c, b), a)
        ]

        return min(candidates)

# Example usage:
# sol = Solution()
# print(sol.minimumString("abc", "bca", "aaa"))  # Output: "aaabca"
# print(sol.minimumString("ab", "ba", "aba"))    # Output: "aba"