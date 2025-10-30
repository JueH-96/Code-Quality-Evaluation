class Solution:
    def minimumCost(self, s: str) -> int:
        n = len(s)
        cost = 0
        
        # Traverse the string from the second character to the second last character
        for i in range(1, n):
            if s[i] != s[i - 1]:
                # If characters are different, we can choose to flip either the prefix or the suffix
                # We choose the minimum cost between flipping up to i or from i to the end
                cost += min(i, n - i)
        
        return cost