import math
from typing import List

class Solution:
    def canTraverseAllPairs(self, nums: List[int]) -> bool:
        """
        Determines if all pairs of indices in the array are traversable.
        Traversal between i and j is possible if gcd(nums[i], nums[j]) > 1.
        This problem can be modeled as finding if the graph of indices is connected.

        The approach uses a Disjoint Set Union (DSU) data structure combined with
        prime factorization to efficiently determine connectivity.
        """
        if len(nums) <= 1:
            return True

        # Using a set is an O(n) operation that helps handle duplicates
        # and provides fast lookups.
        unique_nums = set(nums)
        
        # If 1 is present and the array has more than one element, it's impossible
        # to connect the index of 1 to any other index, since gcd(1, x) = 1.
        if 1 in unique_nums:
            return False
        
        max_val = max(unique_nums)

        # Sieve to find the Smallest Prime Factor (SPF) for all numbers up to max_val.
        # This precomputation allows for efficient prime factorization later.
        spf = list(range(max_val + 1))
        for i in range(2, int(math.sqrt(max_val)) + 1):
            if spf[i] == i:  # i is a prime number
                for j in range(i * i, max_val + 1, i):
                    if spf[j] == j:  # Update only if not already updated by a smaller prime
                        spf[j] = i

        # DSU data structure implemented with path compression and union by size.
        # The DSU operates on numbers up to max_val.
        parent = list(range(max_val + 1))
        size = [1] * (max_val + 1)
        
        def find(i):
            if parent[i] == i:
                return i
            parent[i] = find(parent[i])
            return parent[i]

        def union(i, j):
            root_i = find(i)
            root_j = find(j)
            if root_i != root_j:
                # Union by size
                if size[root_i] < size[root_j]:
                    root_i, root_j = root_j, root_i
                parent[root_j] = root_i
                size[root_i] += size[root_j]

        # For each number, union it with its prime factors.
        # This connects numbers that share common prime factors.
        for num in unique_nums:
            temp_num = num
            while temp_num > 1:
                prime_factor = spf[temp_num]
                union(num, prime_factor)
                # Efficiently divide by the prime factor
                while temp_num % prime_factor == 0:
                    temp_num //= prime_factor
        
        # Check if all numbers in unique_nums belong to the same component.
        # If they do, it means all original numbers in nums are mutually connected.
        first_num_root = find(next(iter(unique_nums)))
        for num in unique_nums:
            if find(num) != first_num_root:
                return False
                
        return True