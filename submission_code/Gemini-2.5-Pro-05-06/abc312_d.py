import sys

def solve():
    S = sys.stdin.readline().strip()
    N = len(S)
    MOD = 998244353

    # If N is odd, it's impossible to form a parenthesis string.
    if N % 2 != 0:
        print(0)
        return

    # dp_current[j] stores the number of ways to form a prefix of a certain length
    # (managed by char_len_idx) ending with balance j.
    # Initially, for a prefix of length 0 (empty string):
    dp_current = [0] * (N + 1)
    dp_current[0] = 1  # Balance 0, 1 way (the empty prefix)

    # char_len_idx represents the length of the prefix for which dp_current holds counts.
    # It iterates from 0 up to N-1.
    # In each iteration, S[char_len_idx] is processed.
    # dp_next will store counts for prefixes of length char_len_idx + 1.
    for char_len_idx in range(N):
        s_char = S[char_len_idx]
        dp_next = [0] * (N + 1)
        
        # Iterate over possible balances j for prefixes of length char_len_idx.
        # Max balance for length char_len_idx is char_len_idx.
        # Balance j must have the same parity as char_len_idx.
        for j in range(char_len_idx % 2, char_len_idx + 1, 2):
            if dp_current[j] == 0:
                continue
            
            current_ways = dp_current[j]

            # Option 1: s_char (S[char_len_idx]) becomes '('
            if s_char == '(' or s_char == '?':
                # New balance is j+1 for length char_len_idx + 1.
                # This new balance cannot exceed N (max possible balance for any prefix).
                # Also, j+1 cannot exceed the new length char_len_idx + 1.
                if j + 1 <= N: 
                    dp_next[j+1] = (dp_next[j+1] + current_ways) % MOD
            
            # Option 2: s_char (S[char_len_idx]) becomes ')'
            if s_char == ')' or s_char == '?':
                # New balance is j-1 for length char_len_idx + 1.
                # This new balance must not be negative.
                if j - 1 >= 0:
                    dp_next[j-1] = (dp_next[j-1] + current_ways) % MOD
        
        dp_current = dp_next # Move to next length

    # After processing all N characters, dp_current holds counts for length N.
    # The answer is the number of ways to achieve balance 0.
    print(dp_current[0])

solve()