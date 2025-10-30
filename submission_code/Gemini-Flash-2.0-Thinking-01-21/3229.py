class Solution:
    def minimumCost(self, nums: List[int]) -> int:
        def is_palindrome(n):
            return str(n) == str(n)[::-1]

        def generate_palindromes(limit):
            palindromes = []
            for i in range(1, 10):
                palindromes.append(i)
            for i in range(10, limit):
                if is_palindrome(i):
                    palindromes.append(i)
            return palindromes

        palindromes = generate_palindromes(10**9)
        min_cost = float('inf')

        for p in palindromes:
            current_cost = 0
            for num in nums:
                current_cost += abs(num - p)
            min_cost = min(min_cost, current_cost)
        return min_cost