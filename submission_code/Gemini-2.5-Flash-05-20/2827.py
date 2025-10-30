import math
from typing import List

# DSU Helper Class (defined outside for clarity and reusability)
class DSU:
    def __init__(self, n):
        # parent[i] stores the parent of element i
        # Initially, each element is its own parent (in its own set)
        self.parent = list(range(n))

    def find(self, i):
        # Path compression: if i is not its own parent,
        # recursively find its root and set it as i's direct parent.
        if self.parent[i] == i:
            return i
        self.parent[i] = self.find(self.parent[i])
        return self.parent[i]

    def union(self, i, j):
        # Find the roots of i and j
        root_i = self.find(i)
        root_j = self.find(j)

        # If they are already in the same set, do nothing
        if root_i != root_j:
            # Union by setting one root's parent to the other root.
            # For competitive programming, simple union by parent modification
            # with path compression is usually sufficient.
            self.parent[root_i] = root_j
            return True
        return False

class Solution:
    # Max value constraint for nums[i] as per problem constraints.
    _MAX_NUM_VAL = 10**5
    
    # Precompute smallest prime factor (SPF) for numbers up to _MAX_NUM_VAL.
    # _spf[x] will store the smallest prime factor of x.
    _spf = [0] * (_MAX_NUM_VAL + 1)
    
    # List to store prime numbers found during sieve, useful for the sieve itself.
    _primes_list = []
    
    # Flag to ensure sieve is initialized only once across multiple test cases.
    _sieve_initialized = False

    @classmethod
    def _sieve_init(cls):
        """
        Initializes the SPF array using a sieve method.
        This method is called once to precompute prime factors.
        """
        if cls._sieve_initialized:
            return

        # Sieve of Eratosthenes to find smallest prime factor (SPF) for each number.
        # 0 and 1 are not prime, their SPF remains 0 (or can be treated specially).
        for i in range(2, cls._MAX_NUM_VAL + 1):
            if cls._spf[i] == 0: # If spf[i] is still 0, it means 'i' is a prime number.
                cls._spf[i] = i # The smallest prime factor of a prime is itself.
                cls._primes_list.append(i) # Add it to our list of primes.
            
            # For each prime 'p' found so far:
            # Mark multiples of 'p' (specifically, i*p) with 'p' as their SPF.
            for p in cls._primes_list:
                # Break conditions:
                # 1. If 'p' is greater than the smallest prime factor of 'i' (cls._spf[i]),
                #    then 'i*p' would already have a smaller prime factor (cls._spf[i]) marked.
                # 2. If 'i*p' exceeds the _MAX_NUM_VAL, no need to process further as it's out of range.
                if p > cls._spf[i] or i * p > cls._MAX_NUM_VAL:
                    break
                cls._spf[i * p] = p # 'p' is the smallest prime factor of 'i*p'.
        
        cls._sieve_initialized = True

    def canTraverseAllPairs(self, nums: List[int]) -> bool:
        n = len(nums)

        # Base Case 1: If there's only one element, there are no pairs to check.
        # It's vacuously true that all pairs can be traversed.
        if n == 1:
            return True

        # Base Case 2: Check for the presence of 1.
        # gcd(1, x) = 1 for any x. This means an index 'i' where nums[i] = 1
        # cannot form a direct edge with any other index 'j'.
        # Consequently, if there's more than one element (n > 1) and one of them is 1,
        # it's impossible to connect all elements because the '1' node will be isolated.
        if 1 in nums:
            return False

        # Initialize the Sieve for SPF if it hasn't been initialized yet.
        # This ensures precomputation is done only once across calls/test cases.
        Solution._sieve_init()

        # DSU setup:
        # We need nodes in our DSU to represent two types of entities:
        # 1. The 'n' indices of the input array (from 0 to n-1).
        # 2. All possible prime factors that can appear in nums (up to _MAX_NUM_VAL).
        #    We map a prime number 'p' to an integer ID `n + p`.
        # The total number of elements in the DSU will be `n + _MAX_NUM_VAL + 1`.
        # For example, if n=10^5 and _MAX_NUM_VAL=10^5, DSU size is approx 2 * 10^5.
        dsu = DSU(n + Solution._MAX_NUM_VAL + 1)

        # Iterate through each number in nums along with its index.
        for i in range(n):
            num = nums[i]
            temp_num = num # Use a temporary variable for factorization to preserve original num.
            
            # Factorize temp_num using the precomputed SPF array.
            # Continue as long as temp_num has factors left (i.e., > 1).
            while temp_num > 1:
                # Get the smallest prime factor of the current temp_num.
                p = Solution._spf[temp_num]
                
                # Union the current index `i` (node ID 'i')
                # with the "node" representing prime `p` (node ID `n + p`).
                # This connects `i` to all its prime factors.
                dsu.union(i, n + p) 
                
                # Divide out all occurrences of this prime factor 'p' from temp_num.
                # This ensures we move to the next distinct prime factor for factorization.
                while temp_num % p == 0:
                    temp_num //= p
        
        # After processing all numbers and performing unions based on common prime factors,
        # we need to check if all original indices (0 to n-1) belong to the same connected component.
        # This can be done by verifying that their roots (as determined by DSU's find operation)
        # are all identical to the root of the first index (index 0).
        root_of_first_element = dsu.find(0)
        for i in range(1, n): # Start from 1 as 0 is our reference.
            if dsu.find(i) != root_of_first_element:
                # If any index 'i' is not connected to index 0, the graph is not fully connected.
                return False 
        
        # If the loop completes, it means all indices are in the same connected component.
        return True