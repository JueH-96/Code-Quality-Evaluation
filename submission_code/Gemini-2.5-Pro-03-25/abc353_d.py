# YOUR CODE HERE
import sys

def solve():
    # Read the number of elements N
    N = int(sys.stdin.readline())
    
    # Read the sequence A of N positive integers
    # Using list comprehension for potentially better performance than map + list conversion
    A = [int(x) for x in sys.stdin.readline().split()]
    
    # Define the modulus
    MOD = 998244353
    
    # Initialize the total sum and the current prefix sum
    total_sum = 0
    current_prefix_sum = 0 # Stores sum A[0] + ... + A[k-1] % MOD

    # Iterate through the sequence A with index k from 0 to N-1
    for k in range(N): 
        # The element being processed in this iteration is A[k].
        
        # The problem asks for the sum of f(A_i, A_j) for all 1 <= i < j <= N.
        # In 0-based indexing, this is Sum_{0 <= i < k <= N-1} f(A[i], A[k]).
        
        # When k > 0, we calculate the contribution to the total sum from pairs
        # where A[k] is the second element (the 'y' in f(x, y)).
        # The total contribution for a fixed k as the second element is Sum_{i=0}^{k-1} f(A[i], A[k]).
        if k > 0: 
             # Determine the number of decimal digits in A[k].
             # The function f(x, y) = x * 10^{d_y} + y, where d_y is the number of digits in y.
             # Here, y is A[k]. We need d_k, the number of digits in A[k].
             num_digits_k = len(str(A[k])) 
             
             # Calculate 10^{d_k} modulo MOD. This factor shifts A[i] to the left
             # appropriately when concatenated with A[k].
             power_of_10 = pow(10, num_digits_k, MOD)
             
             # The sum Sum_{i=0}^{k-1} f(A[i], A[k]) can be expanded as:
             # Sum_{i=0}^{k-1} (A[i] * 10^{d_k} + A[k])
             # This can be split into two parts based on the summation properties:
             # Part 1: (Sum_{i=0}^{k-1} A[i]) * 10^{d_k}
             # Part 2: Sum_{i=0}^{k-1} A[k] = k * A[k]
             
             # Calculate Part 1: (prefix sum up to k-1) * 10^{d_k}
             # `current_prefix_sum` variable holds Sum_{i=0}^{k-1} A[i] % MOD.
             term1 = (current_prefix_sum * power_of_10) % MOD
             
             # Calculate Part 2: k * A[k]
             # Note that k is the count of elements before A[k] (indices 0 to k-1).
             term2 = (k * A[k]) % MOD
             
             # Combine Part 1 and Part 2 modulo MOD. This is the total contribution
             # for pairs ending with A[k].
             term_k = (term1 + term2) % MOD
             
             # Add this contribution to the overall total sum, maintaining the result modulo MOD.
             total_sum = (total_sum + term_k) % MOD

        # Update the prefix sum to include the current element A[k].
        # This updated value will be used in the next iteration (k+1) as Sum_{i=0}^{k} A[i].
        current_prefix_sum = (current_prefix_sum + A[k]) % MOD
        
    # After iterating through all elements, `total_sum` holds the final result modulo MOD.
    # Print the final answer.
    print(total_sum)

# Execute the solve function to run the program
solve()