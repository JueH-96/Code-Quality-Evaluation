class Solution:
    def minimumCost(self, s: str) -> int:
        n = len(s)
        # Initialize costs for making all characters '0' or '1'
        cost_to_all_zero = cost_to_all_one = 0

        # Count the number of '1's and '0's
        count_one = s.count('1')
        count_zero = n - count_one

        # Calculate the cost to make all characters '0'
        for i in range(n):
            if s[i] == '1':
                cost_to_all_zero += i + 1

        # Calculate the cost to make all characters '1'
        for i in range(n):
            if s[i] == '0':
                cost_to_all_one += n - i

        # Return the minimum cost
        return min(cost_to_all_zero, cost_to_all_one)