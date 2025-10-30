from math import gcd
from typing import List

class Solution:
    def canTraverseAllPairs(self, nums: List[int]) -> bool:
        if len(nums) == 1:
            return True
        if 1 in nums:
            return False
        
        max_num = max(nums)
        sieve = [True] * (max_num + 1)
        primes = []
        
        for i in range(2, max_num + 1):
            if sieve[i]:
                primes.append(i)
                for j in range(i * i, max_num + 1, i):
                    sieve[j] = False
        
        graph = {}
        for num in nums:
            factors = set()
            for prime in primes:
                if prime * prime > num:
                    break
                if num % prime == 0:
                    factors.add(prime)
                    while num % prime == 0:
                        num //= prime
            if num > 1:
                factors.add(num)
            for factor in factors:
                if factor not in graph:
                    graph[factor] = set()
                graph[factor].add(num)
                if num not in graph:
                    graph[num] = set()
                graph[num].add(factor)
        
        visited = set()
        stack = [nums[0]]
        
        while stack:
            node = stack.pop()
            if node in visited:
                continue
            visited.add(node)
            for neighbor in graph[node]:
                if neighbor not in visited:
                    stack.append(neighbor)
        
        return len(visited) == len(nums)