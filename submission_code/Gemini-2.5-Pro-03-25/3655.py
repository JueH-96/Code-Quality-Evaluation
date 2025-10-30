import heapq
import math

# --- Start of Global Precomputation ---
# Precompute primes using Sieve of Eratosthenes up to MAX_VAL.
# This global computation ensures the sieve is run only once, 
# even if the Solution class is instantiated multiple times.
MAX_VAL = 10000
is_prime = [True] * MAX_VAL
# 0 and 1 are not prime numbers.
is_prime[0] = is_prime[1] = False
# Calculate primes up to sqrt(MAX_VAL) for optimization.
limit = int(math.sqrt(MAX_VAL)) + 1
for p in range(2, limit):
    # If p is prime (i.e., is_prime[p] is still True)
    if is_prime[p]:
        # Mark all multiples of p as not prime.
        # Start from p*p because smaller multiples (like 2*p, 3*p, ...) 
        # would have already been marked by smaller primes (2, 3, ...).
        for i in range(p * p, MAX_VAL, p):
            is_prime[i] = False
# --- End of Global Precomputation ---

class Solution:
    def minOperations(self, n: int, m: int) -> int:
        """
        Calculates the minimum cost to transform integer n into integer m
        using digit increment/decrement operations, ensuring that the integer 
        represented by n is never a prime number at any point during the transformation.
        The cost of a transformation is the sum of all values that n takes 
        throughout the operations performed, including the starting value n and
        the ending value m.

        Args:
            n: The starting integer.
            m: The target integer.

        Returns:
            The minimum cost to transform n into m, or -1 if it's impossible.
        """
        
        # Access the globally precomputed prime list.
        # This allows the method to efficiently check for primality.
        global is_prime, MAX_VAL

        # --- Initial Checks ---
        # The problem statement implies n and m are within the valid range [1, 9999].
        # We rely on the precomputed is_prime list which covers this range.

        # If the starting number 'n' or the target number 'm' is prime, 
        # the transformation is impossible according to the rules.
        if is_prime[n] or is_prime[m]:
            return -1
            
        # Trivial case: If n is already equal to m.
        # Since n is not prime (checked above), the cost is just the value of n itself,
        # as it's the only number in the sequence.
        if n == m:
            return n

        # --- Dijkstra's Algorithm Setup ---
        
        # Determine the number of digits for n and m. The problem guarantees 
        # they have the same number of digits, and this number must be maintained.
        num_digits = len(str(n))
        
        # 'costs' dictionary stores the minimum cost found so far to reach each node (number).
        # The cost is defined as the sum of node values along the path.
        # Initialize the cost for the starting node 'n'.
        costs = {n: n} 
        
        # Priority queue (min-heap) stores tuples of (current_cost, current_node).
        # It helps Dijkstra's algorithm efficiently select the next node to visit
        # based on the minimum accumulated cost. Initialize with the starting node.
        pq = [(n, n)] # (cost_to_reach_n, n)

        # --- Dijkstra's Algorithm Execution ---
        while pq:
            # Extract the node 'u' with the smallest cost 'current_cost' from the priority queue.
            current_cost, u = heapq.heappop(pq)

            # If we have reached the target node 'm', we have found the path with 
            # the minimum cost according to Dijkstra's algorithm. Return the cost.
            if u == m:
                 return current_cost 

            # Optimization: If the cost popped from the priority queue ('current_cost') 
            # is greater than the already recorded minimum cost to reach 'u' (costs[u]),
            # it means we found a shorter path to 'u' earlier and already processed it.
            # This entry is outdated, so skip processing to avoid redundant work.
            if current_cost > costs[u]:
                 continue

            # --- Generate Neighbors (Potential Next States) ---
            # Convert the current number 'u' to its string representation.
            # Use zfill to pad with leading zeros if necessary, ensuring it always has 'num_digits'.
            # This simplifies digit manipulation at fixed positions.
            s = str(u).zfill(num_digits) 

            # Iterate through each digit position of the number.
            for i in range(num_digits):
                digit = int(s[i])

                # --- Explore neighbor by INCREASING the digit ---
                # Check if the current digit can be increased (must be less than 9).
                if digit < 9:
                    # Construct the string representation of the potential neighbor 'v'.
                    new_s = s[:i] + str(digit + 1) + s[i+1:]
                    # Convert the string back to an integer.
                    v = int(new_s)
                    
                    # Check validity of the neighbor 'v':
                    # 1. It must not be a prime number.
                    # 2. It must be within the precomputed range [0, MAX_VAL-1]. (Implicitly true here).
                    if not is_prime[v]:
                        # Calculate the cost to reach 'v' through the current path ending at 'u'.
                        # The cost definition adds the value of the node 'v' itself.
                        new_cost_v = current_cost + v 
                        
                        # If 'v' has not been visited before, or if this new path to 'v' 
                        # has a lower cost than the previously recorded best cost for 'v':
                        if v not in costs or new_cost_v < costs[v]:
                            # Update the minimum cost to reach 'v'.
                            costs[v] = new_cost_v
                            # Add 'v' to the priority queue with its new cost for further exploration.
                            heapq.heappush(pq, (new_cost_v, v))

                # --- Explore neighbor by DECREASING the digit ---
                # Check if the current digit can be decreased (must be greater than 0).
                if digit > 0:
                    # Apply the constraint: cannot decrease the leading digit (i=0) to '0' 
                    # if the number has more than one digit, as this would change the number of digits.
                    if i == 0 and digit == 1 and num_digits > 1:
                        continue # Skip this operation as it's invalid.
                    
                    # Construct the string representation of the potential neighbor 'v'.
                    new_s = s[:i] + str(digit - 1) + s[i+1:]
                    # Convert the string back to an integer.
                    v = int(new_s)
                     
                    # Check validity of the neighbor 'v': Must not be prime.
                    if not is_prime[v]:
                        # Calculate the cost to reach 'v' through the current path.
                        new_cost_v = current_cost + v
                        
                        # If this path to 'v' is better than any previously known path:
                        if v not in costs or new_cost_v < costs[v]:
                            # Update the minimum cost to reach 'v'.
                            costs[v] = new_cost_v
                            # Add 'v' to the priority queue.
                            heapq.heappush(pq, (new_cost_v, v))

        # If the priority queue becomes empty and we haven't reached 'm', 
        # it means 'm' is unreachable from 'n' while satisfying all constraints 
        # (non-prime intermediates, fixed number of digits).
        return -1