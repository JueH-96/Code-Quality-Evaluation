class Solution:
    def minimumCost(self, nums: List[int]) -> int:
        def is_palindromic(x):
            return str(x) == str(x)[::-1]
        
        # Generate all palindromic numbers less than 10^9
        palindromic_numbers = []
        for i in range(1, 10**9):
            if is_palindromic(i):
                palindromic_numbers.append(i)
            if i > 10**6:  # Optimization to stop early
                break
        
        min_cost = float('inf')
        
        for palindromic in palindromic_numbers:
            current_cost = sum(abs(num - palindromic) for num in nums)
            min_cost = min(min_cost, current_cost)
        
        return min_cost