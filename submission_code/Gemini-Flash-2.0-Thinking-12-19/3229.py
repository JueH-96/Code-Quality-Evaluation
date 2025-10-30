class Solution:
    def minimumCost(self, nums: List[int]) -> int:
        def is_palindrome(n):
            return str(n) == str(n)[::-1]

        def generate_palindromes(limit):
            palindromes = set()
            for i in range(1, 100000):
                s = str(i)
                p1 = int(s + s[::-1])
                if p1 < limit:
                    palindromes.add(p1)
                p2 = int(s + s[:-1][::-1])
                if p2 < limit:
                    palindromes.add(p2)
            for i in range(1, 10):
                palindromes.add(i)
            return sorted(list(palindromes))

        palindromes = generate_palindromes(10**9)
        min_cost = float('inf')

        for pal in palindromes:
            cost = sum(abs(num - pal) for num in nums)
            min_cost = min(min_cost, cost)

        return min_cost