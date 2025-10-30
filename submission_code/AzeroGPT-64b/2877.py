class Solution:
    def minimumString(self, a: str, b: str, c: str) -> str:
        def merge(s1, s2):
            # For each possible suffix of s1, check if it's a prefix of s2 and update the minimum required length
            for i in range(len(s1) - 1, -1, -1):
                if s1[i:] == s2[: len(s1) - i]:
                    return s1 + s2[len(s1) - i :]

            # If there is no overlap, just concatenate the two strings
            return s1 + s2

        candidates = [merge(merge(a, b), c), merge(merge(b, a), c), merge(merge(a, c), b), merge(merge(c, a), b), merge(merge(b, c), a), merge(merge(c, b), a)]

        # Sort for lexicographically smallest order
        candidates.sort(key=len)
        candidates_with_min_length = [candidate for candidate in candidates if len(candidate) == len(candidates[0])]

        # Find the lexicographically smallest string among those with the minimum length
        return min(candidates_with_min_length)