class Solution:
    def minimumOperations(self, num: str) -> int:
        n = len(num)
        
        # A number is special if it's divisible by 25.
        # This means it must end in "00", "25", "50", "75", or be 0.

        # Baseline answer:
        # We can delete all digits to get 0. Cost = n.
        # If num contains a '0', we can delete n-1 digits to get the number 0.
        # So, the answer is at most n, and at most n-1 if a '0' is present.
        min_ops = n if '0' not in num else n - 1

        def find_cost_for_ending(ending: str) -> int:
            """
            Calculates the minimum operations to make num end with the given two-digit string.
            Returns a large value (n) if impossible.
            """
            d1, d2 = ending[0], ending[1]
            
            # Find the rightmost occurrence of the last digit d2
            j = num.rfind(d2)
            if j == -1:
                return n

            # Find the rightmost occurrence of the first digit d1 to the left of d2
            i = num.rfind(d1, 0, j)
            if i == -1:
                return n

            # The number of deletions is the sum of digits between i and j, and digits after j.
            # Deletions = (j - i - 1) + (n - 1 - j)
            # This simplifies to: n - i - 2
            return n - i - 2
        
        # Calculate minimum operations for each possible 2-digit ending
        cost_00 = find_cost_for_ending("00")
        cost_25 = find_cost_for_ending("25")
        cost_50 = find_cost_for_ending("50")
        cost_75 = find_cost_for_ending("75")
        
        # The final answer is the minimum of all possibilities.
        return min(min_ops, cost_00, cost_25, cost_50, cost_75)