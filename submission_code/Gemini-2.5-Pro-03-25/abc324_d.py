# YOUR CODE HERE
import sys
import math
from collections import Counter

# Function to read input line more efficiently
def R(): return sys.stdin.readline().strip()

# Main solver function
def solve():
    # Read the length of the string N
    N = int(R())
    # Read the string S consisting of digits
    S = R()
    
    # Precompute the frequency count of each digit ('0' through '9') in the input string S.
    # Using collections.Counter provides a convenient way to store and compare digit frequencies.
    S_counts = Counter(S)
    
    # Calculate the maximum value of k such that k^2 could potentially be formed.
    # A number formed by permuting N digits can have a value up to 10^N - 1.
    # For example, if N=3, the largest number could be 999.
    # Any square number k^2 formed must therefore satisfy k^2 < 10^N.
    # This implies k < sqrt(10^N).
    # The largest integer k we need to check is floor(sqrt(10^N - 1)).
    
    # Calculate 10^N. Using integer exponentiation is precise.
    limit = 10**N
    
    # Calculate max_k = floor(sqrt(limit - 1)).
    # `int(math.sqrt(x))` correctly computes floor(sqrt(x)) for non-negative x.
    # Handles edge case N=1 where limit=10, limit-1=9, sqrt(9)=3, int(3)=3. Max k is 3.
    if limit == 1: 
         # This case theoretically shouldn't happen since N >= 1. Included for safety.
         max_k = 0 
    else:
         max_k = int(math.sqrt(limit - 1))

    # Initialize count of distinct square numbers found
    count = 0
    
    # Iterate through all possible integer bases k from 0 up to max_k.
    for k in range(max_k + 1): 
        # Calculate the square number
        k_squared = k * k
        
        # Convert the square number k^2 to its string representation T
        T = str(k_squared)
        # Get the number of digits M in k^2
        M = len(T)
        
        # The number of digits M must be at most N.
        # Since k^2 < 10^N, k^2 will always have at most N digits.
        # Example: N=4, k^2 < 10000. Max k^2 is 99^2 = 9801 (4 digits). M <= N is guaranteed.
        # Therefore, an explicit check like `if M > N:` is not necessary here.
        
        # Calculate the required multiset of digits for k^2 to be formed from S.
        # If k^2 has M digits, it must correspond to an N-digit permutation of S.
        # This means the string representation of k^2 must effectively be padded with N-M leading zeros.
        # The total N digits used (T's digits + padding zeros) must match the digits in S.
        
        # Calculate the frequency count of digits in T (the string form of k^2).
        T_counts = Counter(T)
        
        # Determine the number of leading zeros needed for padding.
        padding_zeros = N - M
        
        # Construct the target frequency count map (`required_counts`).
        # This map represents the multiset of digits in the N-digit string that evaluates to k^2.
        # For example, if N=3, k^2=1, T="1", M=1. Padding zeros = 2.
        # Required counts should represent "001": {'0': 2, '1': 1}.
        required_counts = T_counts.copy()
        # Update the count for digit '0' by adding the number of padding zeros.
        # Use .get('0', 0) to handle cases where T initially doesn't contain '0'.
        required_counts['0'] = required_counts.get('0', 0) + padding_zeros
        
        # Check if the multiset of digits in S (S_counts) matches the required multiset (required_counts).
        # Counter objects equality comparison efficiently checks if both multisets are identical.
        if S_counts == required_counts:
            # If they match, k^2 is a square number that can be formed by a permutation of S.
            count += 1

    # Print the final total count of distinct square numbers found.
    print(count)

# Call the solver function to execute the program logic
solve()