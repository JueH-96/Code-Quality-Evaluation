class Solution:
    def minChanges(self, s: str) -> int:
        n = len(s)
        changes = 0
        # Process the string in pairs
        for i in range(0, n, 2):
            # Count cost if we want the pair to be '00'
            cost_00 = (s[i] != '0') + (s[i+1] != '0')
            # Count cost if we want the pair to be '11'
            cost_11 = (s[i] != '1') + (s[i+1] != '1')
            # Take the minimum cost for this pair
            changes += min(cost_00, cost_11)
        return changes