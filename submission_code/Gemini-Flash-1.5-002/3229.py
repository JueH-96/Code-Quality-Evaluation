class Solution:
    def minimumCost(self, nums: List[int]) -> int:
        def is_palindrome(n):
            return str(n) == str(n)[::-1]

        palindromes = [x for x in range(1, 1000) if is_palindrome(x)]
        min_cost = float('inf')

        for p in palindromes:
            cost = 0
            for num in nums:
                cost += abs(num - p)
            min_cost = min(min_cost, cost)
        
        
        return min_cost