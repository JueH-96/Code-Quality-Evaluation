# YOUR CODE HERE
MOD = 998244353

def count_good_strings(N, K, S):
    # Precompute all possible masks for the first K characters
    # Each mask represents a sequence of A and B, where A is 0 and B is 1
    # We need to ensure that no window of size K is a palindrome
    # A window is a palindrome if the i-th character equals the (K-1-i)-th character for all i
    
    # Generate all possible masks for the first K characters
    # Since K is up to 10, 2^10 is 1024, which is manageable
    possible_masks = []
    for mask in range(0, 2**K):
        # Check if the mask represents a palindrome
        is_palindrome = True
        for i in range(K):
            if ((mask >> i) & 1) != ((mask >> (K-1-i)) & 1):
                is_palindrome = False
                break
        if not is_palindrome:
            possible_masks.append(mask)
    
    # Now, we need to count the number of ways to fill the ?s such that no window of size K is a palindrome
    # We can use dynamic programming to count the valid sequences
    
    # Initialize the DP table
    # dp[i][mask] represents the number of ways to fill the first i characters, where the last K characters are represented by mask
    # mask is a bitmask representing the last K characters
    # Since K can be up to 10, we need to manage the mask efficiently
    
    # Initialize the DP table for the first K characters
    dp = {}
    for mask in possible_masks:
        # Check if the mask is compatible with the first K characters of S
        compatible = True
        for i in range(K):
            char = S[i]
            if char == 'A' and ((mask >> i) & 1) != 0:
                compatible = False
                break
            if char == 'B' and ((mask >> i) & 1) != 1:
                compatible = False
                break
        if compatible:
            dp[mask] = 1
    
    # Iterate through the remaining characters
    for i in range(K, N):
        new_dp = {}
        for mask in dp:
            # Get the last K-1 characters
            last_k_minus_1 = mask >> 1
            # Try to add A (0) and B (1)
            for new_char in [0, 1]:
                # Check if the new character is compatible with S[i]
                char = S[i]
                if char == 'A' and new_char != 0:
                    continue
                if char == 'B' and new_char != 1:
                    continue
                # Form the new mask
                new_mask = (last_k_minus_1 << 1) | new_char
                # Check if the new mask is a palindrome
                is_palindrome = True
                for j in range(K):
                    if ((new_mask >> j) & 1) != ((new_mask >> (K-1-j)) & 1):
                        is_palindrome = False
                        break
                if not is_palindrome:
                    if new_mask in new_dp:
                        new_dp[new_mask] = (new_dp[new_mask] + dp[mask]) % MOD
                    else:
                        new_dp[new_mask] = dp[mask] % MOD
        dp = new_dp
    
    # Sum all the valid ways
    total = 0
    for mask in dp:
        total = (total + dp[mask]) % MOD
    return total

# Read input
N, K = map(int, input().split())
S = input().strip()

# Compute the answer
result = count_good_strings(N, K, S)

# Print the result
print(result)