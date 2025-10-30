import sys

# Setting a high recursion depth typical for competitive programming problems
# involving deep recursion. Python's default limit (often 1000) might be
# insufficient for large N (up to 10^5).
# m = (n+1)//2 can be up to ~50000. Need depth > m.
try:
    # Set recursion depth slightly larger than the maximum possible m.
    # Using 100000 + 50 as a safe margin for N up to 10^5.
    sys.setrecursionlimit(100000 + 50) 
except Exception as e:
    # In some environments (like restricted online judges), setting recursion
    # depth might fail. We proceed without it, but might fail for large N.
    # Avoid printing warnings in the final submission code.
    pass 

class Solution:
    """
    Finds the largest n-digit integer that is a palindrome and divisible by k.
    
    The approach uses dynamic programming with memoization. We try to construct 
    the first half of the palindrome (let its length be m = ceil(n/2)) greedily 
    from the most significant digit to the least significant. For each position, 
    we try digits from 9 down to 0 (or 1 for the first digit), and recursively 
    find the largest possible suffix that satisfies the divisibility constraint modulo k.
    The divisibility check is done using precomputed weights based on powers of 10 modulo k.
    """
    def largestPalindrome(self, n: int, k: int) -> str:
        """
        Args:
          n: The number of digits the integer must have (1 <= n <= 10^5).
          k: The divisor (1 <= k <= 9).

        Returns:
          The largest n-digit k-palindromic integer as a string.
          Returns "-1" if no such integer exists (though the problem implies one always does
          for the given constraints, this handles potential edge cases like n=1, k=7).
        """

        # Optimization: If k=1, any number is divisible by 1.
        # The largest n-digit palindrome is simply "9" repeated n times.
        if k == 1:
            return "9" * n

        # Handle the base case n=1 separately for clarity.
        if n == 1:
            # Find the largest single digit (1-9) divisible by k.
            # Digits must be positive since leading zeros are not allowed.
            for d in range(9, 0, -1):
                if d % k == 0:
                    return str(d)
            # If no digit from 1 to 9 is divisible by k (e.g., k=7),
            # then no 1-digit k-palindromic number exists.
            return "-1" 

        # Calculate the length of the first half of the palindrome.
        # m = ceil(n/2)
        m = (n + 1) // 2

        # Precompute powers of 10 modulo k: pow10[i] = (10^i) % k.
        # Needed up to exponent n.
        pow10 = [1] * (n + 1)
        for i in range(1, n + 1):
            pow10[i] = (pow10[i - 1] * 10) % k

        # Precompute the weights W[i] for each digit d_i in the first half (1-based index).
        # The palindrome x can be expressed as sum_{i=1}^{m} d_i * W'_i for some weights W'_i.
        # The condition x % k == 0 translates to (sum_{i=1}^{m} d_i * W[i]) % k == 0,
        # where W[i] are the weights modulo k.
        W = [0] * (m + 1)  # Use 1-based indexing: W[1]...W[m]
        
        if n % 2 == 1: # Odd number of digits (n = 2m - 1)
            # The digit d_i at position i contributes d_i * 10^(n-i).
            # Its symmetric digit d_{n-i+1} is also d_i and contributes d_i * 10^(n-(n-i+1)) = d_i * 10^(i-1).
            for i in range(1, m): # For digits i = 1 to m-1
                W[i] = (pow10[n - i] + pow10[i - 1]) % k
            # The middle digit d_m (at position m = n-m+1) contributes d_m * 10^(n-m).
            W[m] = pow10[n - m] % k  # Note: n - m = m - 1 for odd n.
        else: # Even number of digits (n = 2m)
             # The digit d_i at position i contributes d_i * 10^(n-i).
             # Its symmetric digit d_{n-i+1} is also d_i and contributes d_i * 10^(i-1).
            for i in range(1, m + 1): # For digits i = 1 to m
                W[i] = (pow10[n - i] + pow10[i - 1]) % k

        # Memoization table for the dynamic programming state: (index, target_mod)
        # Stores the result of solve(i, target_mod) to avoid recomputation.
        memo = {}

        def solve(i, target_mod):
            """
            Recursive function with memoization (top-down DP).
            Finds the lexicographically largest suffix d_i...d_m such that 
            the weighted sum (d_i*W[i] + ... + d_m*W[m]) % k equals target_mod.
            
            Args:
              i: The current digit index (from 1 to m).
              target_mod: The required value of the weighted sum modulo k for the suffix d_i...d_m.

            Returns:
              The lexicographically largest suffix string if found, otherwise None.
            """
            # Base case: If we have determined all m digits (i goes from 1 to m).
            if i == m + 1:
                # If the target modulo is 0, it means the chosen digits satisfy the condition.
                # Return an empty string representing a valid completion.
                # If target modulo is non-zero, this path is invalid.
                return "" if target_mod == 0 else None
            
            # Create state tuple for memoization lookup.
            state = (i, target_mod)
            
            # If this state has already been computed, return the stored result.
            if state in memo:
                return memo[state]

            # Determine the range of digits 'd' to try for the current position 'i'.
            # We iterate from 9 down to 0 (or 1) to find the largest digit first.
            start_digit = 9
            end_digit = -1 # Loop iterates from start_digit down to end_digit + 1 (exclusive)
            if i == 1: # The first digit (d_1) cannot be 0.
                 end_digit = 0 # Loop range becomes [9, 1], i.e., 9 down to 1.

            # Try digits d for position i from largest to smallest.
            for d in range(start_digit, end_digit, -1):
                # Calculate the required modulo sum for the rest of the suffix (i+1 ... m).
                # We need: (d * W[i] + sum_{j=i+1}^{m} d_j * W[j]) % k == target_mod
                # So, the recursive call needs to find a suffix whose sum is:
                # sum_{j=i+1}^{m} d_j * W[j] % k == (target_mod - d * W[i]) % k
                remaining_target_mod = (target_mod - d * W[i]) % k
                # Python's % operator handles negative results correctly, 
                # e.g., (-1) % 6 = 5. No need for manual adjustment to non-negative.
                
                # Recursively call solve for the next position (i+1) with the new target modulo.
                suffix = solve(i + 1, remaining_target_mod)
                
                # If the recursive call found a valid suffix:
                if suffix is not None:
                    # We found the largest possible digit 'd' for position 'i' that leads to a solution.
                    # Construct the result for the current state by prepending 'd'.
                    result = str(d) + suffix
                    # Store the result in the memo table.
                    memo[state] = result
                    # Return the found result.
                    return result
            
            # If no digit 'd' for the current position 'i' leads to a valid solution.
            # Store None in the memo table to indicate impossibility for this state.
            memo[state] = None
            return None

        # Start the DP calculation to find the first half (prefix) p_str.
        # We need the total weighted sum to be 0 modulo k for the full palindrome.
        p_str = solve(1, 0)

        # If p_str is None, it means no n-digit k-palindromic number exists.
        if p_str is None:
             return "-1" 

        # Construct the full palindrome string from the first half (p_str).
        if n % 2 == 1: # Odd length n: The middle character of p_str is the center.
            # Example: n=5, m=3. p_str="abc". Palindrome="abcba".
            # We need p_str + reverse(p_str without the last character).
            prefix_part = p_str[:-1] # The part before the middle char, e.g., "ab".
            palindrome_str = p_str + prefix_part[::-1] # Concatenate: "abc" + "ba".
        else: # Even length n: The palindrome is formed by p_str and its reverse.
            # Example: n=6, m=3. p_str="abc". Palindrome="abccba".
            # We need p_str + reverse(p_str).
            palindrome_str = p_str + p_str[::-1] # Concatenate: "abc" + "cba".
            
        return palindrome_str