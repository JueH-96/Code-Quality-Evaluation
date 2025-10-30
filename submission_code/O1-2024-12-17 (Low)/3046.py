class Solution:
    def minimumOperations(self, num: str) -> int:
        n = len(num)
        
        # If we can reduce the number to a single '0', that is also divisible by 25.
        # The cost for that is n - 1 (remove all other digits) if there's at least one '0'.
        cost_single_zero = float('inf')
        if '0' in num:
            cost_single_zero = n - 1
        
        # Function to find the cost of making the string end with two given digits p1p2
        # by deleting characters in between or after.
        def cost_two_digits(p1, p2):
            # We want to find from right to left the position i such that num[i] == p2,
            # and then from right to left before i, a position j such that num[j] == p1.
            # cost = (n-1 - i) + (i-1 - j)
            # If not found, return a large cost.
            i = -1
            for idx in range(n - 1, -1, -1):
                if num[idx] == p2:
                    i = idx
                    break
            if i == -1:
                return float('inf')
            
            j = -1
            for idx in range(i - 1, -1, -1):
                if num[idx] == p1:
                    j = idx
                    break
            if j == -1:
                return float('inf')
            
            return (n - 1 - i) + (i - 1 - j)
        
        # We'll try all endings "00", "25", "50", "75"
        patterns = ["00", "25", "50", "75"]
        min_cost = float('inf')
        for pattern in patterns:
            p1, p2 = pattern[0], pattern[1]
            min_cost = min(min_cost, cost_two_digits(p1, p2))
        
        # Compare with the cost of just keeping a single '0' if possible
        min_cost = min(min_cost, cost_single_zero)
        
        return min_cost