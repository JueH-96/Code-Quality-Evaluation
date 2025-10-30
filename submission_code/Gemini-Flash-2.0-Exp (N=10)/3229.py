from typing import List

class Solution:
    def minimumCost(self, nums: List[int]) -> int:
        def is_palindrome(n):
            return str(n) == str(n)[::-1]

        palindromes = []
        for i in range(1, 100000):
            if is_palindrome(i):
                palindromes.append(i)
        for i in range(100000, 1000000):
            if is_palindrome(i):
                palindromes.append(i)
        for i in range(1000000, 10000000):
            if is_palindrome(i):
                palindromes.append(i)
        for i in range(10000000, 100000000):
            if is_palindrome(i):
                palindromes.append(i)
        for i in range(100000000, 1000000000):
            if is_palindrome(i):
                palindromes.append(i)
        
        min_cost = float('inf')
        for p in palindromes:
            cost = 0
            for num in nums:
                cost += abs(num - p)
            min_cost = min(min_cost, cost)
        
        return min_cost