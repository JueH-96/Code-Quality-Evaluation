import collections
import heapq

# Global sieve for primality test, precomputed once
MAX_VAL = 9999
_is_prime_sieve = [True] * (MAX_VAL + 1)
_sieve_computed = False

def _compute_sieve():
    """
    Computes prime numbers up to MAX_VAL using the Sieve of Eratosthenes.
    This function is designed to be called only once globally.
    """
    global _sieve_computed
    if not _sieve_computed:
        _is_prime_sieve[0] = _is_prime_sieve[1] = False # 0 and 1 are not prime
        for p in range(2, int(MAX_VAL**0.5) + 1):
            if _is_prime_sieve[p]:
                for multiple in range(p*p, MAX_VAL + 1, p):
                    _is_prime_sieve[multiple] = False
        _sieve_computed = True

class Solution:
    def minOperations(self, n: int, m: int) -> int:
        _compute_sieve() # Ensure sieve is computed before starting operations

        # Initial and target values must not be prime numbers
        if _is_prime_sieve[n] or _is_prime_sieve[m]:
            return -1

        # Store the original number of digits, as all intermediate values must maintain this.
        original_n_len = len(str(n))

        # dist_array[i] stores the minimum total cost to reach integer i
        # Initialize with infinity for all unreachable states.
        dist_array = [float('inf')] * (MAX_VAL + 1)

        # Priority queue for Dijkstra's algorithm: (current_accumulated_cost, current_value)
        # We start at 'n', and its cost is 'n' itself (as per problem definition:
        # "sum of all values that n takes throughout the operations performed").
        pq = [(n, n)]
        dist_array[n] = n

        while pq:
            current_cost, current_val = heapq.heappop(pq)

            # If we've already found a cheaper path to current_val, skip this one
            if current_cost > dist_array[current_val]:
                continue

            # If current_val is our target 'm', we've found the minimum cost
            if current_val == m:
                return current_cost

            s_val = str(current_val)
            
            # Explore possible next states by changing one digit at a time
            for i in range(len(s_val)):
                digit = int(s_val[i])
                
                # Operation: Increase digit
                if digit < 9: # Cannot increase a 9
                    temp_list = list(s_val)
                    temp_list[i] = str(digit + 1)
                    next_val_str = "".join(temp_list)
                    next_val = int(next_val_str)

                    # Check validity of the new number:
                    # 1. Must maintain the same number of digits as the original 'n'.
                    #    Example: if n=10 (2 digits), 19->20 is valid. But 9->19 is not if original n was 9 (1 digit).
                    # 2. Must be positive (as per constraints 1 <= n, m).
                    # 3. Must not be a prime number.
                    if (len(str(next_val)) == original_n_len and 
                        next_val > 0 and 
                        not _is_prime_sieve[next_val]):
                        
                        new_total_cost = current_cost + next_val
                        if new_total_cost < dist_array[next_val]:
                            dist_array[next_val] = new_total_cost
                            heapq.heappush(pq, (new_total_cost, next_val))

                # Operation: Decrease digit
                if digit > 0: # Cannot decrease a 0
                    temp_list = list(s_val)
                    temp_list[i] = str(digit - 1)
                    next_val_str = "".join(temp_list)
                    next_val = int(next_val_str)

                    # Apply the same validity checks as for the increase operation
                    if (len(str(next_val)) == original_n_len and 
                        next_val > 0 and 
                        not _is_prime_sieve[next_val]):
                        
                        new_total_cost = current_cost + next_val
                        if new_total_cost < dist_array[next_val]:
                            dist_array[next_val] = new_total_cost
                            heapq.heappush(pq, (new_total_cost, next_val))
        
        # If the loop finishes and 'm' was not reached, it's impossible
        return -1