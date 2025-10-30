class Solution:
    def minimumCost(self, s: str) -> int:
        n = len(s)
        cost = 0

        # Count the number of '1's in the string
        count_ones = s.count('1')

        # If all characters are already the same, no cost is needed
        if count_ones == 0 or count_ones == n:
            return 0

        # Initialize the minimum cost to a large value
        min_cost = float('inf')

        # Try all possible indices i and calculate the cost
        for i in range(n):
            # Cost to make all characters from 0 to i equal to '1'
            cost_to_make_ones = i + 1 - s[:i+1].count('1')
            # Cost to make all characters from i to n-1 equal to '0'
            cost_to_make_zeros = (n - i) - (s[i:].count('0'))

            # Total cost for this index
            total_cost = cost_to_make_ones + cost_to_make_zeros

            # Update the minimum cost
            min_cost = min(min_cost, total_cost)

        return min_cost