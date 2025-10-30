# YOUR CODE HERE
import sys
import math

# Standard input reading function
def readline():
    return sys.stdin.readline().strip()

# GCD function
def gcd(a, b):
    """Computes the greatest common divisor of a and b."""
    while b:
        a, b = b, a % b
    return abs(a)

# Extended Euclidean Algorithm
def extended_gcd(a, b):
    """Computes gcd(a, b) and Bezout coefficients x, y such that ax + by = gcd(a, b)."""
    if a == 0:
        # Base case: gcd(0, b) = b, Bezout's identity: 0*x + 1*y = b
        return b, 0, 1
    # Recursive step
    d, x1, y1 = extended_gcd(b % a, a)
    # Compute coefficients for gcd(a, b) using results from gcd(b % a, a)
    x = y1 - (b // a) * x1
    y = x1
    return d, x, y

# Modular Inverse function
def modInverse(a, m):
    """Computes the modular multiplicative inverse of a modulo m."""
    d, x, y = extended_gcd(a, m)
    if d != 1:
        # Modular inverse exists only if gcd(a, m) == 1
        return None 
    # Return the inverse modulo m, ensuring it's positive
    return (x % m + m) % m

# Counts number of solutions to ax = b (mod m) for x in [0, n-1]
def count_congruence(a, b, m, n):
    """Counts the number of integers x in [0, n-1] such that ax = b (mod m)."""
    if m <= 0: raise ValueError("m must be positive")
    if n <= 0: return 0

    # Reduce a and b modulo m
    a = a % m
    b = b % m
    
    # Compute gcd(a, m)
    g = gcd(a, m)

    # Check if a solution exists
    if b % g != 0:
        return 0 # No solution if b is not divisible by gcd(a, m)

    # Reduce the congruence equation by dividing by g
    a_red = a // g
    b_red = b // g
    m_red = m // g

    # Find one solution x0 to a_red * x = b_red (mod m_red)
    # Since gcd(a_red, m_red) = 1, modular inverse exists
    a_inv = modInverse(a_red, m_red)
    if a_inv is None: # Should not happen because gcd is 1
         raise RuntimeError("Modular inverse failed unexpectedly.")

    # Calculate the first non-negative solution
    x0 = (b_red * a_inv) % m_red
    
    # General solutions are x = x0 + k*m_red for k=0, 1, ..., g-1.
    # These are g distinct solutions modulo m = m_red * g.
    # We need to count how many of these solutions fall within the range [0, n-1]
    
    if x0 >= n: # The first solution is already outside the range
        return 0
    
    # Count solutions x0 + k*m_red < n
    # k*m_red <= n - 1 - x0
    # k <= floor((n - 1 - x0) / m_red)
    
    max_k = (n - 1 - x0) // m_red
    
    # Number of valid k values is the count of integers in [0, g-1] that are also <= max_k
    # This is count of k in [0, min(g-1, max_k)]
    # The number of integers in [0, K] is K+1.
    count = min(g - 1, max_k) + 1
    
    return count


# Verified floor_sum implementation for non-negative a, b.
# Based on the implementation from `ac-library-python`.
def floor_sum_non_negative_verified(n, m, a, b):
    """Calculates sum_{i=0}^{n-1} floor((a * i + b) / m) for non-negative a, b."""
    ans = 0
    while True:
        if a >= m:
            # Reduce a using identity floor((a*i+b)/m) = floor(((a%m)*i + b)/m) + (a//m)*i
            ans += (n - 1) * n // 2 * (a // m)
            a %= m
        if b >= m:
            # Reduce b using identity floor((a*i+b)/m) = floor((a*i + b%m)/m) + (b//m)
            ans += n * (b // m)
            b %= m
        
        # Calculate maximum value of floor term
        y_max = (a * n + b) // m
        if y_max == 0:
            # If maximum value is 0, all terms are 0. We are done.
            break
        
        # The core transformation step based on properties of floor function sums
        # Relates S(n, m, a, b) to S(y_max, a, m, ...)
        x = y_max * m - b # A term used in the transformation
        ans += (n - (x + a - 1) // a) * y_max # Add part of the sum computed directly
        
        # Update parameters for the recursive call / next iteration
        # New 'b' based on remainder properties
        b = (a - x % a) % a 
        # Swap roles of variables
        n, m, a = y_max, a, m
    return ans

# Wrapper function for floor_sum that handles potentially negative a and b.
def floor_sum_wrapper(n, m, a, b):
    """Calculates sum_{i=0}^{n-1} floor((a * i + b) / m) for any integer a, b and positive integer m."""
    if n <= 0: return 0
    if m <= 0: raise ValueError("m must be positive")
    
    ans_shift = 0 # Accumulates shifts from handling negative b
    
    # Handle potentially negative b by shifting it to non-negative range [0, m-1)
    if b < 0:
        # Calculate k = ceil(-b/m)
        k = (-b + m - 1) // m 
        # Adjust the sum: Each floor term floor((ai+b)/m) effectively decreases by k
        ans_shift = -n * k
        # Update b to be non-negative: b' = b + k*m
        b += k * m
    
    # Now b >= 0. Handle potentially negative a
    if a < 0:
        a_abs = -a
        # Use the identity relating floor sum with negative coefficient to positive coefficient form:
        # Sum floor((a*i+b)/m) = - Sum ceil((a_abs*i - b)/m)
        # And ceil(Z) = floor(Z) + [Z not integer]
        # Sum = - Sum floor((a_abs*i - b)/m) - Sum [a_abs*i != b mod m]
        # Sum = - floor_sum_wrapper(n, m, a_abs, -b) - (n - count_congruence(a_abs, b % m, m, n))
        
        # Recursive call handles the potentially negative argument '-b'
        term1 = floor_sum_wrapper(n, m, a_abs, -b) 
        
        # Count solutions to a_abs * i = b (mod m) for i in [0, n-1]
        num_congr = count_congruence(a_abs, b % m, m, n) # Use current b which is >= 0
        
        # Combine results, including the initial shift for negative b
        return ans_shift - term1 - (n - num_congr)

    # Base case: a >= 0 and b >= 0
    # Call the verified implementation for non-negative arguments
    # Add the initial shift obtained from handling negative b if any
    return ans_shift + floor_sum_non_negative_verified(n, m, a, b)


# Solve function using the interval-based approach
def solve_interval():
    """Solves one test case using the interval-based summation approach."""
    N = int(readline())
    constraints = []
    for i in range(N):
        # Store constraints as [Ai, Bi, Ci, original_index]
        constraints.append(list(map(int, readline().split())) + [i]) 

    # Calculate Y_max = min_i floor( (C_i - 1 - A_i) / B_i ) and check if (1, 1) is possible
    Y_max = (10**18) # Use a large number as initial infinity
    possible = True
    
    for i in range(N):
        Ai, Bi, Ci, _ = constraints[i]
        # Check if (1, 1) is a valid point: A_i*1 + B_i*1 < C_i  <=>  A_i + B_i <= C_i - 1
        if Ci - 1 < Ai + Bi:
           possible = False # If (1,1) is not valid, no positive integer pair (x,y) can be valid.
           break
        
        # Update Y_max based on the constraint for x=1
        # Requires y <= floor( (C_i - 1 - A_i) / B_i )
        if Bi <= 0: # Constraints state Bi >= 1, so this case won't happen
            pass 
        else: # Bi > 0
           numerator = Ci - 1 - Ai
           if numerator < 0: 
               # If numerator < 0, then floor(negative / positive) <= -1.
               # This means y cannot even be 1 for this constraint.
               cur_Y_max = -1 
           else:
              cur_Y_max = numerator // Bi # Integer division is floor
           Y_max = min(Y_max, cur_Y_max)
           
    # If (1,1) is impossible or Y_max calculation shows no y>=1 possible, answer is 0.
    if not possible or Y_max < 1:
        print(0)
        return

    total_sum = 0
    curr_y = 1
    
    # Iterate through intervals of y where k(y) is constant
    while curr_y <= Y_max:
        # Compute K = k(curr_y) = min_i floor( (Ci - 1 - Bi * curr_y) / Ai )
        # This K represents the maximum x for the current y.
        K = (10**18) # Effective infinity for min calculation
        
        possible_k = True
        for i in range(N):
            Ai, Bi, Ci, _ = constraints[i]
            
            numerator = Ci - 1 - Bi * curr_y
            if Ai <= 0: # Constraints state Ai >= 1, so this case won't happen
                 pass
            else: # Ai > 0
                 if numerator < 0: 
                     # If numerator < 0, floor is negative. Since x must be >= 1, k(curr_y) <= 0.
                     K = 0 # Assign 0 or negative, loop will break.
                     possible_k = False
                     break 
                 
                 floor_val = numerator // Ai
                 K = min(K, floor_val)
        
        # If K is non-positive, no valid x >= 1 exists for this y.
        if not possible_k or K <= 0: 
            break # Since k(y) is non-increasing, no more positive K for larger y.

        # Compute Y_next = min_i floor( (Ci - 1 - K * Ai) / Bi )
        # This is the largest y' such that k(y') >= K.
        Y_next = (10**18) # Effective infinity
        possible_y_next = True
        for i in range(N):
             Ai, Bi, Ci, _ = constraints[i]
             numerator = Ci - 1 - K * Ai
             if Bi <= 0: # constraints Bi >= 1
                  pass
             else: # Bi > 0
                 if numerator < 0:
                     # This implies k(y) < K for all y >= 1 for this constraint i.
                     # This means the K derived earlier is impossible over the whole range.
                     # Should signal an issue or means Y_next is essentially before start.
                     Y_next = curr_y - 1 # Set Y_next to force termination of range.
                     possible_y_next = False
                     break 
                 
                 floor_val = numerator // Bi
                 Y_next = min(Y_next, floor_val)
        
        # If Y_next calculation resulted in value less than current y, something is inconsistent.
        # This could mean the determined K is not achievable for curr_y or beyond. Stop safely.
        if not possible_y_next or Y_next < curr_y:
             break 

        # Ensure Y_next does not exceed the overall maximum possible Y_max
        Y_next = min(Y_next, Y_max)
        
        # Calculate number of y values in the current constant interval [curr_y, Y_next]
        count = Y_next - curr_y + 1
        
        # Add contribution K * count to total sum
        # This corresponds to summing K for each y in the interval.
        total_sum += K * count
        
        # Move to the next y value to start checking the next interval
        curr_y = Y_next + 1

    print(total_sum)

# Read number of test cases
T = int(readline())
# Set higher recursion depth for potentially deep floor_sum calls
# sys.setrecursionlimit(2000) 

# Process each test case
for _ in range(T):
    solve_interval()