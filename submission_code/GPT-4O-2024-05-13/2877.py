class Solution:
    def minimumString(self, a: str, b: str, c: str) -> str:
        from itertools import permutations

        def merge_two_strings(x, y):
            # Try to merge y into x
            for i in range(len(x)):
                if x[i:] == y[:len(x) - i]:
                    return x + y[len(x) - i:]
            return x + y

        def merge_three_strings(x, y, z):
            return merge_two_strings(merge_two_strings(x, y), z)

        # Generate all permutations of the three strings
        candidates = []
        for perm in permutations([a, b, c]):
            merged_string = merge_three_strings(perm[0], perm[1], perm[2])
            candidates.append(merged_string)

        # Find the minimum length string, and if there are ties, the lexicographically smallest one
        min_length = min(len(s) for s in candidates)
        min_candidates = [s for s in candidates if len(s) == min_length]
        return min(min_candidates)

# Example usage:
# sol = Solution()
# print(sol.minimumString("abc", "bca", "aaa"))  # Output: "aaabca"
# print(sol.minimumString("ab", "ba", "aba"))    # Output: "aba"