import math # This import is not strictly needed for the final solution but was considered during thought process.

class Solution:
    def valueAfterKSeconds(self, n: int, k: int) -> int:
        """
        Calculates the value of the last element of an array after k seconds.

        The array 'a' starts as [1, 1, ..., 1] of size n.
        After each second, each element a[i] is simultaneously updated to become 
        the sum of elements a[0]...a[i] from the state at the previous second.
        The calculation is performed modulo 10^9 + 7.

        Args:
          n: The size of the array (1 <= n <= 1000).
          k: The number of seconds (1 <= k <= 1000).

        Returns:
          The value of a[n-1] after k seconds, modulo 10^9 + 7.
        
        Approach:
        We can simulate the process described. Let the array at second 's' be a_s.
        a_0 = [1, 1, ..., 1]
        a_{s+1}[i] = sum(a_s[j] for j from 0 to i) % MOD
        
        We observe that the update rule for a_{s+1}[i] corresponds to calculating prefix sums.
        a_{s+1}[i] = (a_{s+1}[i-1] + a_s[i]) % MOD for i > 0, and a_{s+1}[0] = a_s[0].
        
        We can simulate this process for k seconds. An efficient way to compute the next state 
        is to iterate through the array and maintain a running prefix sum. The update can be 
        done in-place because the calculation of the new value for a[i] depends on the prefix sum 
        up to that point, which correctly uses the values from the *previous* state before they 
        are overwritten within the current iteration.

        Example Trace (n=4, k=1):
        Initial a = [1, 1, 1, 1] (state s=0)
        Simulate for k=1 (calculate state s=1):
          prefix_sum = 0
          i=0: ps = (0 + a[0]_0) % MOD = 1. Update a[0] = 1. Array is [1, 1, 1, 1].
          i=1: ps = (1 + a[1]_0) % MOD = 2. Update a[1] = 2. Array is [1, 2, 1, 1].
          i=2: ps = (2 + a[2]_0) % MOD = 3. Update a[2] = 3. Array is [1, 2, 3, 1].
          i=3: ps = (3 + a[3]_0) % MOD = 4. Update a[3] = 4. Array is [1, 2, 3, 4].
        Final state after 1 second: a = [1, 2, 3, 4].
        
        The simulation takes O(k * n) time, which is feasible for n, k <= 1000.
        The space complexity is O(n) for the array.
        
        Alternatively, one could derive a combinatorial formula: a_k[n-1] = C(n + k - 1, k) % MOD.
        This requires calculating binomial coefficients modulo a prime, typically involving 
        precomputation of factorials and modular inverses. The time complexity would be O(n+k) 
        for precomputation and O(1) for the final calculation. While asymptotically faster, 
        the simulation approach is simpler to implement correctly for the given constraints.
        """
        MOD = 10**9 + 7

        # Initialize the array 'a' with n ones.
        # Using a list in Python allows for efficient modification.
        a = [1] * n

        # Simulate the process for k seconds.
        for _ in range(k):
            # In each second, update the array based on the prefix sum rule.
            # a_new[i] = sum(a_old[0]...a_old[i]) % MOD
            # This can be implemented efficiently using a running prefix sum.
            prefix_sum = 0
            for i in range(n):
                 # Calculate the prefix sum up to index i using values from the previous state.
                 # The current a[i] holds the value from the previous second until it's updated.
                 prefix_sum = (prefix_sum + a[i]) % MOD
                 # Update a[i] to its value for the next second (current state being computed).
                 a[i] = prefix_sum

        # After k seconds, the array 'a' contains the final state.
        # Return the value of the last element, a[n-1].
        # The value is guaranteed to be modulo MOD due to the operations inside the loop.
        return a[n - 1]