def solve_problem():
    MOD = 998244353
    
    # Read the input
    N = int(input())
    
    # Calculate the number of digits for each number from 1 to N
    digit_count = [len(str(i)) for i in range(N + 1)]
    
    # Precompute factorials
    factorial = [1] * (N + 1)
    for i in range(1, N + 1):
        factorial[i] = (factorial[i - 1] * i) % MOD
    
    # Memoization dictionary for the sum_of_powers function
    memo = {}
    
    def sum_of_powers(S, size):
        """
        Compute the sum of 10^(sum of digits of subset) for all subsets of given size from the set S.
        """
        if (S, size) in memo:
            return memo[(S, size)]
        
        if size == 0:
            return 1  # Only the empty subset
        
        if len(S) == 0:
            return 0  # No subset of size > 0 from an empty set
        
        # Include the first element or exclude it
        first, rest = S[0], S[1:]
        new_rest = tuple(rest)  # Convert to tuple for consistency
        
        include = (pow(10, digit_count[first], MOD) * sum_of_powers(new_rest, size - 1)) % MOD
        exclude = sum_of_powers(new_rest, size)
        
        result = (include + exclude) % MOD
        memo[(S, size)] = result
        return result
    
    total_sum = 0
    
    # For each number j (1 to N)
    for j in range(1, N + 1):
        numbers_excluding_j = tuple(i for i in range(1, N + 1) if i != j)
        
        # For each position p (0 to N-1)
        for p in range(N):
            subset_size = N - 1 - p
            
            power_sum = sum_of_powers(numbers_excluding_j, subset_size)
            
            # Number of permutations with j at position p
            # and a specific subset after it
            num_perms = (factorial[subset_size] * factorial[p]) % MOD
            
            # Contribution of j at position p
            j_contribution = (j * power_sum % MOD * num_perms) % MOD
            
            total_sum = (total_sum + j_contribution) % MOD
    
    return total_sum

# Run the solver
print(solve_problem())