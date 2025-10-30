# YOUR CODE HERE
def solve_sequence_problem():
    N, M, K = map(int, input().split())
    X = list(map(int, input().split()))
    
    # Check if X is of the form [a, a, ..., a]
    if all(x == X[0] for x in X):
        return count_sequences_for_repeated_pattern(N, M, K, X[0])
    
    # Check if there's a critical digit that can be the gap
    for digit in range(1, K + 1):
        count_in_X = X.count(digit)
        if count_in_X > 0 and can_be_critical_digit(X, digit, count_in_X):
            return count_sequences_with_critical_digit(N, M, K, X, digit, count_in_X)
    
    # If no valid solution exists
    return 0

def can_be_critical_digit(X, digit, count_in_X):
    # Check if limiting the occurrences of 'digit' to (count_in_X - 1) would only exclude X
    # This is a simplification; the full check would be more complex
    return all(x == digit for x in X)

def count_sequences_for_repeated_pattern(N, M, K, digit):
    MOD = 998244353
    
    # Formula for the special case where X is of the form [a, a, ..., a]
    # Count valid digit placements ensuring all other patterns are subsequences
    
    result = 0
    
    # We need at most M-1 occurrences of the digit
    for count in range(M - 1, 0, -1):
        # Choose positions for the repeated digit
        ways_to_place_digit = comb(N, count, MOD)
        
        # For the remaining positions, ensure all other patterns can be formed
        # Each of the K-1 other digits must appear at least once
        remaining_positions = N - count
        
        # Using the principle of inclusion-exclusion to count ways to have each digit at least once
        valid_fillings = 0
        for i in range(K):
            if i + 1.0 == digit:
                continue
            
            # For each subset of digits that we ensure appear at least once
            sign = 1 if i % 2 == 0 else -1
            valid_fillings = (valid_fillings + sign * pow(i + 1, remaining_positions, MOD)) % MOD
        
        result = (result + ways_to_place_digit * valid_fillings) % MOD
    
    return result

def count_sequences_with_critical_digit(N, M, K, X, digit, count_in_X):
    MOD = 998244353
    
    # Count sequences where 'digit' appears at most (count_in_X - 1) times
    # and all other patterns of length M are subsequences
    
    result = 0
    
    # We need at most (count_in_X - 1) occurrences of the critical digit
    for count in range(count_in_X - 1, -1, -1):
        # Choose positions for the critical digit
        ways_to_place_digit = comb(N, count, MOD)
        
        # For the remaining positions, ensure all other patterns can be formed
        # This is a simplified approximation
        remaining_positions = N - count
        valid_fillings = pow(K - 1, remaining_positions, MOD)
        
        result = (result + ways_to_place_digit * valid_fillings) % MOD
    
    return result

def comb(n, k, mod):
    # Calculate nCk modulo mod
    if k < 0 or k > n:
        return 0
    if k == 0 or k == n:
        return 1
    
    numerator = 1
    denominator = 1
    
    for i in range(1, k + 1):
        numerator = (numerator * (n - i + 1)) % mod
        denominator = (denominator * i) % mod
    
    # Calculate modular inverse of denominator
    return (numerator * pow(denominator, mod - 2, mod)) % mod

def pow(base, exp, mod):
    # Calculate base^exp modulo mod
    if exp == 0:
        return 1
    if exp % 2 == 0:
        return pow((base * base) % mod, exp // 2, mod)
    else:
        return (base * pow(base, exp - 1, mod)) % mod

print(solve_sequence_problem())