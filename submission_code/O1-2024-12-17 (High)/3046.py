class Solution:
    def minimumOperations(self, num: str) -> int:
        # We want to make the number (given as a string) end with "00", "25", "50", or "75"
        # by deleting some digits. Also, we can turn the entire number into a single '0'
        # if at least one '0' is present, or remove all digits (which also becomes "0").
        
        # Helper to compute the cost (number of deletions) to form a 2-digit pattern p
        # by matching its last two characters from right to left in num.
        def cost_to_form_pattern(num, p):
            n = len(num)
            d1, d2 = p[0], p[1]
            
            # Find index i for last digit d2 from the right
            i = -1
            for idx in range(n - 1, -1, -1):
                if num[idx] == d2:
                    i = idx
                    break
            if i == -1:
                return float('inf')
            
            # Find index j for second last digit d1, to the left of i
            j = -1
            for idx in range(i - 1, -1, -1):
                if num[idx] == d1:
                    j = idx
                    break
            if j == -1:
                return float('inf')
            
            # Number of digits removed = digits after i + digits between j+1 and i-1
            return (n - 1 - i) + (i - 1 - j)
        
        n = len(num)
        
        # If there's at least one '0', we can keep a single '0' by removing all other digits => cost = n - 1
        # Otherwise, removing all digits => cost = n
        cost_single_zero = (n - 1) if ('0' in num) else n
        
        # Compute cost for each valid 2-digit pattern
        patterns = ["00", "25", "50", "75"]
        best_2digit_cost = float('inf')
        for p in patterns:
            best_2digit_cost = min(best_2digit_cost, cost_to_form_pattern(num, p))
        
        # Removing all digits also yields "0"
        remove_all_cost = n
        
        # The answer is the minimum of forming a single '0', forming "00"/"25"/"50"/"75", or removing all
        return min(cost_single_zero, best_2digit_cost, remove_all_cost)