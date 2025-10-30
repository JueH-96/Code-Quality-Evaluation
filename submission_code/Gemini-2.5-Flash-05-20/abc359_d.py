import sys

def solve():
    N, K = map(int, sys.stdin.readline().split())
    S = sys.stdin.readline().strip()

    MOD = 998244353

    # Function to check if a K-bit integer represents a palindrome
    # Example: if K=4, k_mask=6 (binary 0110), it checks if bit 0 == bit 3 and bit 1 == bit 2.
    # The bits are indexed from right to left (0 being LSB).
    # To get the j-th character from the left (0-indexed) in a K-bit sequence, use (k_mask >> (K - 1 - j)) & 1.
    def is_palindrome(k_mask, K):
        for j in range(K // 2):
            bit_left = (k_mask >> (K - 1 - j)) & 1
            bit_right = (k_mask >> j) & 1
            if bit_left != bit_right:
                return False
        return True

    # num_masks is 2^(K-1), representing all possible states for the last K-1 characters.
    num_masks = 1 << (K - 1)
    
    # dp[i][mask] stores the number of ways to form the prefix S[0...i-1]
    # such that the last K-1 characters are represented by 'mask',
    # and no K-length palindrome has occurred.
    # dp table has (N+1) rows (for prefixes of length 0 to N) and num_masks columns.
    dp = [[0] * num_masks for _ in range(N + 1)]

    # Base case: empty prefix (length 0) has 1 way, with an effective mask of all zeros.
    dp[0][0] = 1

    # Iterate through the string S from left to right (character S[i])
    for i in range(N):
        current_char_s = S[i] # The character at the current position in the input string

        # Iterate through all possible previous masks from dp[i]
        for prev_mask in range(num_masks):
            if dp[i][prev_mask] == 0:
                continue # No ways to reach this state, so skip

            # Try placing 'A' (0) or 'B' (1) at the current position S[i]
            for char_val in range(2): # 0 for 'A', 1 for 'B'
                # If S[i] is a fixed character, ensure char_val matches it.
                if current_char_s != '?' and \
                   ((current_char_s == 'A' and char_val == 1) or \
                    (current_char_s == 'B' and char_val == 0)):
                    continue # Mismatch, this path is invalid

                # Calculate the new mask for dp[i+1].
                # This mask represents the last K-1 characters of the string of length i+1.
                # It's formed by shifting prev_mask left (to make space for new char)
                # and adding char_val, then taking the last K-1 bits.
                new_mask = ((prev_mask << 1) | char_val) & (num_masks - 1)

                # Check for a K-length palindrome.
                # A K-length substring ending at index 'i' (S[i-K+1 ... i]) is formed.
                # This check is only relevant if we have at least K characters in the string,
                # meaning 'i' must be at least K-1.
                
                # The K-bit integer representing the substring S[i-K+1 ... i] is:
                # (bits from prev_mask) followed by char_val.
                k_len_palindrome_candidate = (prev_mask << 1) | char_val

                is_bad_palindrome = False
                if i >= K - 1: # We have enough characters (at least K) to form a K-length substring
                    if is_palindrome(k_len_palindrome_candidate, K):
                        is_bad_palindrome = True
                
                # If the current placement does not form a bad palindrome, add to dp[i+1]
                if not is_bad_palindrome:
                    dp[i+1][new_mask] = (dp[i+1][new_mask] + dp[i][prev_mask]) % MOD

    # The total count of good strings is the sum of all valid ways to form the full string S (length N).
    total_good_strings = 0
    for mask in range(num_masks):
        total_good_strings = (total_good_strings + dp[N][mask]) % MOD

    sys.stdout.write(str(total_good_strings) + '
')

solve()