class Solution:
    def minimumCost(self, nums: List[int]) -> int:
        def is_palindrome(x):
            return str(x) == str(x)[::-1]

        # Generate all palindromic numbers less than 10^9
        palindromic_numbers = set()
        for i in range(1, 10**5):  # 10^5 is enough to generate palindromes < 10^9
            s = str(i)
            palindromic_numbers.add(int(s + s[::-1]))  # Even length
            palindromic_numbers.add(int(s + s[-2::-1]))  # Odd length
        
        # Filter palindromic numbers less than 10^9
        palindromic_numbers = [p for p in palindromic_numbers if p < 10**9]
        
        # Calculate the minimum cost to convert all nums to each palindromic number
        min_cost = float('inf')
        for p in palindromic_numbers:
            cost = sum(abs(num - p) for num in nums)
            min_cost = min(min_cost, cost)
        
        return min_cost