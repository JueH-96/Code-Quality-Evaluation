from typing import List

class Solution:
    def minimumCost(self, nums: List[int]) -> int:
        # Generate all palindromic numbers less than 10^9
        palindromic_numbers = []
        for i in range(1, 10**5):
            str_i = str(i)
            if str_i == str_i[::-1]:
                palindromic_numbers.append(i)
        
        # Initialize minimum cost to infinity
        min_cost = float('inf')
        
        # For each palindromic number, calculate the cost to make all elements in nums equal to it
        for palindromic_number in palindromic_numbers:
            cost = sum(abs(num - palindromic_number) for num in nums)
            # Update minimum cost if the current cost is smaller
            min_cost = min(min_cost, cost)
        
        return min_cost