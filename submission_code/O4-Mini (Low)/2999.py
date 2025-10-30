class Solution:
    def canBeEqual(self, s1: str, s2: str) -> bool:
        # We can only swap positions (0,2) and (1,3) any number of times.
        # Thus the multiset of chars at even indices {0,2} in s1 must match that in s2,
        # and likewise for odd indices {1,3}.
        return (
            sorted((s1[0], s1[2])) == sorted((s2[0], s2[2])) and
            sorted((s1[1], s1[3])) == sorted((s2[1], s2[3]))
        )

# Example usage:
if __name__ == "__main__":
    sol = Solution()
    print(sol.canBeEqual("abcd", "cdab"))  # True
    print(sol.canBeEqual("abcd", "dacb"))  # False