class Solution:
    def minimumCost(self, nums: List[int]) -> int:
        def is_palindrome(n):
            return str(n) == str(n)[::-1]

        palindromes = set()
        for i in range(1, 10):
            palindromes.add(i)

        for i in range(10, 100):
            palindromes.add(int(str(i) + str(i)[::-1]))

        for i in range(1, 1000):
            s = str(i)
            palindromes.add(int(s + s[:-1][::-1]))

        for i in range(10, 10000):
            s = str(i)
            palindromes.add(int(s + s[::-1]))

        for i in range(100, 100000):
            s = str(i)
            palindromes.add(int(s + s[:-1][::-1]))

        for i in range(1000, 1000000):
            s = str(i)
            palindromes.add(int(s + s[::-1]))

        for i in range(10000, 10000000):
            s = str(i)
            palindromes.add(int(s + s[:-1][::-1]))

        for i in range(100000, 100000000):
            s = str(i)
            palindromes.add(int(s + s[::-1]))

        valid_palindromes = sorted([p for p in palindromes if p < 10**9])

        min_cost = float('inf')

        for pal in valid_palindromes:
            current_cost = 0
            for num in nums:
                current_cost += abs(num - pal)
            min_cost = min(min_cost, current_cost)

        return min_cost