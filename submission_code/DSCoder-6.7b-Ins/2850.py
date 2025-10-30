class Solution:
    def longestString(self, x: int, y: int, z: int) -> int:
        # The maximum possible length of the new string is determined by the number of "AA" and "BB" substrings.
        # Since we cannot have "AAA" or "BBB" as a substring, we can only use "AA" and "BB" in pairs.
        # Therefore, the maximum length is 2*min(x, y) + 2*z.
        # If x and y are equal, we can use either "AA" or "BB" in pairs, so the maximum length is 2*min(x, y) + 2*min(x, y) = 4*min(x, y).
        # If x and y are not equal, we can use "AA" in pairs and "BB" in pairs separately, so the maximum length is 2*min(x, y) + 2*z.
        return min(x, y) * 4 + max(x, y) * 2 + 2 * z