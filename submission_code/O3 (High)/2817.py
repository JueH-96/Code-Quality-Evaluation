class Solution:
    def minimumCost(self, s: str) -> int:
        """
        For every position where the character changes with respect to the
        previous one we MUST flip either the left part (a prefix that ends
        at the previous index) or the right part (a suffix that starts at the
        current index) an odd number of times – otherwise the two adjacent
        characters could never become equal.

        • Flipping a prefix that ends at index i-1 costs i.
        • Flipping a suffix that starts at index i     costs n-i.

        In the optimal solution we therefore pay, for every change of
        character, the cheaper of those two amounts.

        Summing that over the whole string gives the minimum total cost.
        """
        n = len(s)
        cost = 0
        for i in range(1, n):
            if s[i] != s[i-1]:                 # boundary where characters differ
                cost += min(i, n - i)          # cheapest way to fix this boundary
        return cost