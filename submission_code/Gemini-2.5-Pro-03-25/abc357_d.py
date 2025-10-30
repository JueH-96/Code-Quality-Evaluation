# YOUR CODE HERE
import sys

# Set higher recursion depth for safety, although iterative modular exponentiation doesn't need it.
# Python's default recursion depth is usually sufficient. If needed, uncomment the line below.
# sys.setrecursionlimit(2000) 

def solve():
    # Read the input integer N from standard input
    N = int(sys.stdin.readline())
    
    # Define the modulus M as specified in the problem
    M = 998244353
    # Define M-1, which is used in applying Fermat's Little Theorem for exponents
    M1 = M - 1

    # Calculate L = number of digits in N.
    # Convert N to string to find its length. This works correctly even for large N.
    sN = str(N)
    L = len(sN)

    # Calculate 10^L mod M using Python's built-in pow function for modular exponentiation.
    # This is the common ratio 'r' in the geometric series.
    TenL = pow(10, L, M)

    # Handle the edge case where the common ratio 10^L is congruent to 1 modulo M.
    # The geometric series formula V_N = N * (r^N - 1) / (r - 1) requires r != 1.
    # If r = 10^L == 1 mod M, the sum becomes V_N = Sum_{i=0}^{N-1} N * (10^L)^i mod M
    # V_N = Sum_{i=0}^{N-1} N * 1^i mod M = Sum_{i=0}^{N-1} N mod M = (N * N) mod M
    if TenL == 1:
        # Compute N mod M first to prevent potential overflow if N*N is very large.
        N_modM = N % M
        # Calculate (N mod M)^2 mod M and print the result.
        print(pow(N_modM, 2, M))
        # Exit the function since we have found the answer for this case.
        return 

    # If 10^L is not congruent to 1 mod M, we can use the standard geometric series sum formula:
    # V_N = N * ( (10^L)^N - 1 ) / (10^L - 1 ) mod M
    # This can be rewritten using modular inverse:
    # V_N = N * (10^(NL) - 1) * modInverse(10^L - 1, M) mod M

    # Compute N mod M needed for the factor N in the formula.
    N_modM = N % M

    # Compute the exponent NL modulo (M-1).
    # We use the property derived from Fermat's Little Theorem: a^b mod M = a^(b mod (M-1)) mod M
    # This holds for prime M and gcd(a, M)=1. Here a=10, M is prime, and 10 is not a multiple of M.
    
    # Compute N mod (M-1) needed for the exponent calculation.
    N_modM1 = N % M1
    
    # Calculate the effective exponent E' = (N mod (M-1)) * L mod (M-1)
    # Python's arbitrary precision integers handle the intermediate product N_modM1 * L correctly.
    E_prime = (N_modM1 * L) % M1

    # Calculate 10^(NL) mod M, which is equivalent to 10^E' mod M.
    TenNL = pow(10, E_prime, M)

    # Calculate the numerator term (10^(NL) - 1) mod M.
    # Add M before taking modulo to ensure the result is non-negative. E.g., if TenNL=1, 1-1=0. If TenNL=0, 0-1=-1 becomes M-1.
    Numerator = (TenNL - 1 + M) % M

    # Calculate the denominator term (10^L - 1) mod M.
    # Add M before taking modulo ensures non-negative result.
    Denominator = (TenL - 1 + M) % M

    # Calculate the modular multiplicative inverse of the denominator modulo M.
    # Since M is prime and Denominator is not 0 mod M (because TenL != 1 mod M),
    # we can use Fermat's Little Theorem: inverse(a) = a^(M-2) mod M.
    DenInv = pow(Denominator, M - 2, M)

    # Final Calculation: (N_modM * Numerator * DenInv) mod M
    
    # Perform multiplications step-by-step, taking modulo M at each step to prevent overflow and keep numbers manageable.
    # Calculate (N_modM * Numerator) mod M first.
    Term1 = (N_modM * Numerator) % M
    
    # Then multiply the result by DenInv mod M.
    Result = (Term1 * DenInv) % M

    # Print the final result.
    print(Result)

# Call the solve function to execute the logic when the script is run.
solve()