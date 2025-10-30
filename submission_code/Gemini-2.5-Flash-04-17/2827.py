from typing import List

class DSU:
    def __init__(self, n):
        # parent[i] stores the parent of element i
        self.parent = list(range(n))
        # size[i] stores the size of the set rooted at i
        self.size = [1] * n

    def find(self, i):
        # Find the root of the set containing element i
        if self.parent[i] == i:
            return i
        # Path compression: Set parent[i] directly to the root
        self.parent[i] = self.find(self.parent[i])
        return self.parent[i]

    def union(self, i, j):
        # Union the sets containing elements i and j
        root_i = self.find(i)
        root_j = self.find(j)
        if root_i != root_j:
            # Union by size: Attach the smaller tree under the root of the larger tree
            if self.size[root_i] < self.size[root_j]:
                root_i, root_j = root_j, root_i
            self.parent[root_j] = root_i
            self.size[root_i] += self.size[root_j]
            return True # Union happened
        return False # Already in the same set

def get_prime_factors(num):
    """Finds the distinct prime factors of a number."""
    factors = set()
    d = 2
    temp = num
    
    # Handle factor 2
    if temp % d == 0:
        factors.add(d)
        while temp % d == 0:
            temp //= d
            
    d = 3
    # Check odd factors from 3 upwards
    while d * d <= temp:
        if temp % d == 0:
            factors.add(d)
            while temp % d == 0:
                temp //= d
        d += 2 # Move to the next odd number
        
    # If temp is still greater than 1, the remaining temp is a prime factor
    if temp > 1:
        factors.add(temp)
        
    return factors


class Solution:
    def canTraverseAllPairs(self, nums: List[int]) -> bool:
        n = len(nums)

        # Edge case: If N = 1, it's vacuously true.
        if n == 1:
            return True
            
        # Edge case: If 1 is present and N > 1, impossible to connect (gcd(1, x) = 1).
        # If all numbers are 1, gcd(1,1)=1, no connections.
        # If some are 1 and some are >1, the index with 1 is isolated from others.
        # So, if 1 in nums and N > 1, return False.
        # Check presence of 1 efficiently.
        if 1 in nums:
             return False

        # Maximum possible value in nums is 10^5.
        # Number of unique prime factors up to 10^5 is pi(10^5) approx 9592.
        # We need DSU nodes for indices (0 to n-1) and for unique prime factors.
        # Prime factor nodes are dynamically assigned IDs starting from n.
        # Max required DSU size = n (for indices) + max_unique_primes_up_to_max_val.
        # Let's use n + 10005 as a safe upper bound for DSU size.
        # The actual number of unique prime factors in the input `nums` is at most pi(10^5).
        dsu = DSU(n + 10005)
        
        # Map prime factor values to their corresponding DSU node IDs
        prime_to_dsu_id = {}
        # Counter for the next available DSU ID for a prime factor
        next_prime_dsu_id = n # Start prime factor IDs after index IDs (0 to n-1)

        # Process each number in nums
        for i in range(n):
            num = nums[i]
            # Get distinct prime factors for the current number
            factors = get_prime_factors(num)

            for p in factors:
                # If this prime factor hasn't been assigned a DSU ID yet
                if p not in prime_to_dsu_id:
                    # Assign a new DSU ID to this prime factor
                    prime_to_dsu_id[p] = next_prime_dsu_id
                    next_prime_dsu_id += 1
                
                # Get the DSU node ID for the prime factor p
                prime_dsu_id = prime_to_dsu_id[p]
                
                # Unite the set containing index i (DSU ID i)
                # and the set containing prime factor p (DSU ID prime_dsu_id).
                # This links index i to all numbers/indices that share the prime factor p.
                # If nums[i] and nums[j] share prime p, then i is united with p, j is united with p.
                # This makes i and j connected via p.
                dsu.union(i, prime_dsu_id)

        # Check if all index nodes (0 to n-1) are in the same set.
        # If they are, it means any index can be reached from any other index.
        # It is sufficient to check if all index nodes have the same root as index 0.
        # We only need to check the first n nodes (representing indices).
        # Problem constraints: 1 <= nums.length, so n >= 1.
        # If n == 1, we already returned True. So here n > 1.
        root_of_first_index = dsu.find(0)

        for i in range(1, n):
            if dsu.find(i) != root_of_first_index:
                return False

        # If the loop completes, all index nodes are in the same set.
        return True