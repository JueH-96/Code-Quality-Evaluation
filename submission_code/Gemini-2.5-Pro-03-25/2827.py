import math
from typing import List

# Disjoint Set Union (DSU) class with path compression and union by size
class DSU:
    """
    A Disjoint Set Union (also known as Union-Find) data structure.
    Supports path compression and union by size for efficiency.
    """
    def __init__(self, n):
        """Initializes the DSU structure for n elements (0 to n-1)."""
        # parent[i] stores the parent of element i. Initially, each element is its own parent.
        self.parent = list(range(n))
        # size[i] stores the size of the set rooted at i. Initially, each set has size 1.
        self.size = [1] * n 
        # Track the number of distinct components (sets). Initially, n components.
        self.num_components = n 

    def find(self, i):
        """Finds the representative (root) of the set containing element i, with path compression."""
        # Base case: if i is the root, return i.
        if self.parent[i] == i:
            return i
        # Recursively find the root and update parent pointer for path compression.
        self.parent[i] = self.find(self.parent[i])
        return self.parent[i]

    def union(self, i, j):
        """
        Unites the sets containing elements i and j using the union by size heuristic.
        Returns True if a merge occurred (i and j were in different sets), False otherwise.
        """
        # Find the representatives (roots) of the sets containing i and j.
        root_i = self.find(i)
        root_j = self.find(j)
        
        # If i and j are already in the same set, do nothing.
        if root_i != root_j:
            # Union by size: attach the smaller tree to the root of the larger tree.
            if self.size[root_i] < self.size[root_j]:
                # Make root_j the parent of root_i.
                self.parent[root_i] = root_j
                # Update the size of the merged set rooted at root_j.
                self.size[root_j] += self.size[root_i]
            else:
                # Make root_i the parent of root_j.
                self.parent[root_j] = root_i
                # Update the size of the merged set rooted at root_i.
                self.size[root_i] += self.size[root_j]
            
            # A merge occurred, so decrease the number of distinct components.
            self.num_components -= 1
            return True # Indicate that a merge happened.
        return False # Indicate that no merge happened.


class Solution:
    """
    This class provides a solution to determine if all pairs of indices in the array `nums` 
    can be traversed, where traversal between index i and j is possible if gcd(nums[i], nums[j]) > 1.
    This problem is equivalent to checking if the graph formed by indices as nodes and 
    edges based on the greatest common divisor (GCD) condition is connected.
    """
    def canTraverseAllPairs(self, nums: List[int]) -> bool:
        """
        Checks if the graph described by the connectivity rule is connected.

        Args:
            nums: A list of 0-indexed integers.

        Returns:
            True if all pairs of indices are traversable (graph is connected), False otherwise.
        """
        n = len(nums)
        if n == 1:
            # A graph with a single node is trivially connected.
            return True

        # Optimization: Check for the presence of 1 in the array.
        # If 1 exists and there are multiple elements (n > 1), 
        # the node corresponding to index k where nums[k]=1 cannot connect to any other node j,
        # because gcd(1, nums[j]) = 1 for any nums[j]. Thus, the graph cannot be fully connected.
        contains_one = False
        for x in nums:
            if x == 1:
                contains_one = True
                break
        
        if contains_one:
             # If 1 is present and N > 1, connectivity is impossible.
             return False
        
        # Find the maximum value in the array to set the range for the Sieve of Eratosthenes.
        # This determines the upper bound for prime factor computation.
        max_val = 0
        for x in nums:
             # Using a loop might be slightly faster than python's built-in max() for very large lists.
             if x > max_val:
                 max_val = x

        # Precompute the Smallest Prime Factor (SPF) for all numbers up to max_val using a sieve.
        # spf[k] will store the smallest prime factor of k. This allows for efficient factorization.
        spf = list(range(max_val + 1))
        # Mark 0 and 1 as having no prime factors in the standard sense.
        spf[0] = spf[1] = -1 
        
        # Sieve implementation: Iterate potential primes up to sqrt(max_val).
        limit = int(math.sqrt(max_val)) + 1
        for i in range(2, limit):
            # If i is prime (its SPF is itself)
            if spf[i] == i: 
                # Mark multiples of i with i as their smallest prime factor.
                # Start marking from i*i because smaller multiples would have already been marked 
                # by smaller prime factors.
                for j in range(i * i, max_val + 1, i):
                    # Update spf[j] only if it hasn't been updated by a smaller prime factor yet.
                    if spf[j] == j: 
                        spf[j] = i
        
        # Initialize the Disjoint Set Union (DSU) data structure for N indices (0 to N-1).
        dsu = DSU(n)
        
        # Dictionary to map each prime factor encountered to the representative index 
        # of the component it is currently associated with.
        prime_to_rep_map = {} 

        # Iterate through each number and its index in the input array.
        for i in range(n):
            num = nums[i]
            # Since we handled the case with 1s earlier, all numbers processed here are > 1.
            
            temp_num = num
            # Factorize num using the precomputed SPF array and perform DSU operations.
            while temp_num > 1:
                # Get the smallest prime factor of the current number.
                prime = spf[temp_num]
                # Check for edge case or error condition.
                if prime == -1: break 
                
                # If this prime factor has been seen before and is associated with a component representative:
                if prime in prime_to_rep_map:
                    # Union the component of the current index 'i' with the component
                    # previously associated with this prime factor. This connects nodes sharing prime factors.
                    dsu.union(i, prime_to_rep_map[prime])
                
                # Update the map: associate this prime factor with the representative 
                # of the current component containing index 'i'. If i was merged into another component,
                # find(i) will return the new representative. This ensures subsequent occurrences
                # of this prime connect to the correct merged component.
                prime_to_rep_map[prime] = dsu.find(i) 
                
                # Efficiently remove all occurrences of this prime factor from temp_num
                # by repeatedly dividing by it, to proceed to the next distinct prime factor.
                while temp_num % prime == 0:
                     temp_num //= prime

        # After processing all numbers and performing unions based on shared prime factors,
        # check if all indices belong to a single connected component.
        # The DSU structure tracks the number of components remaining.
        # If num_components is 1, it means the graph is fully connected.
        return dsu.num_components == 1