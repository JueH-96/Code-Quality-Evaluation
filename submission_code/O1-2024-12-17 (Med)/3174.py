class Solution:
    def minChanges(self, s: str) -> int:
        # We can partition the string into chunks of length 2 (pairs).
        # For each pair, we can make it "00" or "11" with some cost.
        # We choose whichever option yields the smaller cost for that pair
        # and sum these costs over all pairs.
        # This ensures each pair is an even-length (2) substring of identical bits,
        # thus the entire string becomes "beautiful".
        
        n = len(s)
        total_cost = 0
        
        # Process the string in steps of 2
        for i in range(0, n, 2):
            zero_cost = (s[i] == '1') + (s[i+1] == '1')  # cost to make pair into "00"
            one_cost = (s[i] == '0') + (s[i+1] == '0')  # cost to make pair into "11"
            total_cost += min(zero_cost, one_cost)
        
        return total_cost