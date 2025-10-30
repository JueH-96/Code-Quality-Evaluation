class Solution:
    def minChanges(self, s: str) -> int:
        n = len(s)
        # We can ensure the string is "beautiful" by making every pair of consecutive
        # characters (s[2k], s[2k+1]) the same. Each such pair can either be "00" or "11".
        # We choose which one requires fewer changes.
        #
        # For pair (a, b):
        #   cost_to_00 = (a != '0') + (b != '0')
        #   cost_to_11 = (a != '1') + (b != '1')
        #   cost_for_pair = min(cost_to_00, cost_to_11)
        #
        # Summing this minimal cost over all pairs gives us the answer.
        
        changes = 0
        for i in range(0, n, 2):
            a, b = s[i], s[i+1]
            cost_00 = (a != '0') + (b != '0')
            cost_11 = (a != '1') + (b != '1')
            changes += min(cost_00, cost_11)
        
        return changes