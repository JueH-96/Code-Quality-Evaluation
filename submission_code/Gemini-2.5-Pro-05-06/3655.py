import heapq

# Global initialization for Sieve of Eratosthenes
MAX_VAL = 10000
is_prime = [True] * MAX_VAL 

def _initialize_sieve():
    """Helper function to populate the is_prime array."""
    # 0 and 1 are not prime.
    is_prime[0] = is_prime[1] = False 
    # Iterate up to sqrt(MAX_VAL -1) for sieve
    for p in range(2, int(MAX_VAL**0.5) + 1): 
        if is_prime[p]:
            for multiple in range(p*p, MAX_VAL, p):
                is_prime[multiple] = False

_initialize_sieve() # This will run once when the module is loaded.

class Solution:
  
    def minOperations(self, n: int, m: int) -> int:
        # Constraints: 1 <= n, m < 10^4. These are within MAX_VAL range.
        
        # Condition: n must not be a prime number at any point.
        # This includes initial n and final m.
        if is_prime[n] or is_prime[m]:
            return -1
        
        # If n is already m, the sequence of values n takes is just (n).
        # The cost is the sum of these values, which is n.
        if n == m:
            return n

        # Dijkstra's algorithm setup
        # dist_arr[x] stores the minimum sum of precursor states (s_0 + ... + s_{k-1}) 
        # to reach state x = s_k, where s_0 is the initial n.
        # For the starting node n (s_0), this sum is empty, so dist_arr[n] = 0.
        dist_arr = [float('inf')] * MAX_VAL
        dist_arr[n] = 0
        
        # Priority queue stores tuples: (cumulative_cost_of_precursors, node_value)
        pq = [(0, n)]
        
        while pq:
            current_sum_precursors, u = heapq.heappop(pq)
            
            # If a shorter path to u has already been found and processed
            if current_sum_precursors > dist_arr[u]:
                continue
            
            # If u is the target m, we've found the minimal sum of precursors.
            # The total cost includes m itself: (s_0 + ... + s_{k-1}) + s_k.
            # This sum is current_sum_precursors + m.
            if u == m:
                 return current_sum_precursors + m

            s_u = str(u) # Convert current number to string for digit manipulation
            len_s_u = len(s_u)

            # Explore neighbors by changing one digit of u
            for i in range(len_s_u): # i is the index of the digit to change
                original_digit_int = int(s_u[i])
                
                # Option 1: Increase the digit (if it's not 9)
                if original_digit_int < 9:
                    # Construct the new number v
                    temp_s_v_list = list(s_u)
                    temp_s_v_list[i] = str(original_digit_int + 1)
                    v = int("".join(temp_s_v_list))
                    
                    # New state v must not be prime.
                    # v is guaranteed to be < MAX_VAL since u < MAX_VAL.
                    if not is_prime[v]:
                        # Cost to reach v through u: (sum of precursors to u) + u itself
                        # current_sum_precursors is dist_arr[u] here.
                        cost_to_reach_v_via_u = current_sum_precursors + u 
                        if cost_to_reach_v_via_u < dist_arr[v]:
                            dist_arr[v] = cost_to_reach_v_via_u
                            heapq.heappush(pq, (dist_arr[v], v))
                
                # Option 2: Decrease the digit (if it's not 0)
                if original_digit_int > 0:
                    # Critical rule: Preserve number of digits for multi-digit numbers.
                    # Disallow changing the leading digit to '0' if the number has > 1 digit.
                    # e.g., "10" -> "00" (value 0) is disallowed. "123" -> "023" (value 23) is disallowed.
                    # For single-digit numbers (len_s_u == 1), e.g., u=1, this rule does not apply. 
                    # 1 -> 0 is allowed. Resulting 0 is also 1-digit (len(str(0))==1).
                    if i == 0 and original_digit_int == 1 and len_s_u > 1:
                        continue # Skip this disallowed operation
                    
                    # Construct the new number v
                    temp_s_v_list = list(s_u)
                    temp_s_v_list[i] = str(original_digit_int - 1)
                    v = int("".join(temp_s_v_list)) # Handles "0X" -> X, "00" -> 0, "0" -> 0
                    
                    # New state v must not be prime.
                    # v can be 0 if u=1 and it's decreased. is_prime[0] is False, so 0 is a valid non-prime state.
                    if not is_prime[v]:
                        cost_to_reach_v_via_u = current_sum_precursors + u
                        if cost_to_reach_v_via_u < dist_arr[v]:
                            dist_arr[v] = cost_to_reach_v_via_u
                            heapq.heappush(pq, (dist_arr[v], v))
        
        # If the priority queue becomes empty and m has not been reached
        return -1