class Solution:
    def getSmallestString(self, s: str, k: int) -> str:
        # Helper function to compute the cyclic distance between two letters.
        def cost(a: str, b: str) -> int:
            diff = abs(ord(a) - ord(b))
            return min(diff, 26 - diff)
        
        n = len(s)
        result = []
        remaining = k
        
        for i in range(n):
            current = s[i]
            # Try to choose the lexicographically smallest letter for this position.
            # Since we can always leave the character unchanged (cost 0),
            # and we have no further restrictions on future positions (they can be left as is),
            # we just need to ensure our move cost does not exceed the remaining budget.
            for c in "abcdefghijklmnopqrstuvwxyz":
                d = cost(current, c)
                if d <= remaining:
                    # Once we have a valid candidate, break out.
                    result.append(c)
                    remaining -= d
                    break
        return "".join(result)
        
# Sample test cases
if __name__ == '__main__':
    sol = Solution()
    # Example 1:
    s1, k1 = "zbbz", 3
    print(sol.getSmallestString(s1, k1))  # Expected "aaaz"
    
    # Example 2:
    s2, k2 = "xaxcd", 4
    print(sol.getSmallestString(s2, k2))  # Expected "aawcd"
    
    # Example 3:
    s3, k3 = "lol", 0
    print(sol.getSmallestString(s3, k3))  # Expected "lol"