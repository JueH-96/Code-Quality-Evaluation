class Solution:
    def minimumCost(self, nums: List[int]) -> int:
        def is_palindrome(x):
            return str(x) == str(x)[::-1]

        def generate_palindromes(limit):
            palindromes = []
            for i in range(1, 10000):  # Generate palindromes up to 999999999
                s = str(i)
                palindromes.append(int(s + s[::-1]))  # Even length palindromes
                palindromes.append(int(s + s[-2::-1]))  # Odd length palindromes
            return sorted(p for p in palindromes if p < limit)

        palindromes = generate_palindromes(10**9)
        
        def calculate_cost(target):
            return sum(abs(num - target) for num in nums)

        min_cost = float('inf')
        for palindrome in palindromes:
            cost = calculate_cost(palindrome)
            if cost < min_cost:
                min_cost = cost

        return min_cost