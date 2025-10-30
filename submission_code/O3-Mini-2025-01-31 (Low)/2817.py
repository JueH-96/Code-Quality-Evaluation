class Solution:
    def minimumCost(self, s: str) -> int:
        n = len(s)
        total_cost = 0
        # For every transition in the string, add the minimal cost to fix it.
        for i in range(1, n):
            if s[i] != s[i-1]:
                total_cost += min(i, n - i)
        return total_cost

# For local testing:
if __name__ == '__main__':
    sol = Solution()
    print(sol.minimumCost("0011"))    # Expected 2
    print(sol.minimumCost("010101"))  # Expected 9