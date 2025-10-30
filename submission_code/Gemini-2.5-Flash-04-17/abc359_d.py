import sys

# Set recursion depth if needed, though not typical for this DP
# sys.setrecursionlimit(2000)

MOD = 998244353

def solve():
    N, K = map(int, sys.stdin.readline().split())
    S = sys.stdin.readline().strip()

    # dp[mask] stores the number of ways to fill the prefix S[0...i-1]
    # such that the resulting string is valid and the last K characters
    # S[i-K...i-1] correspond to mask.
    # We iterate i from 0 to N. dp[i] is computed from dp[i-1].
    # dp[i] stores counts for S[0...i-1]. Mask represents S[i-K...i-1].
    # Mask: integer where bit j (0 to K-1) represents the character S[i-1-j] (0 for 'A', 1 for 'B').
    # LSB (bit 0) is the character S[i-1], MSB (bit K-1) is S[i-K].

    # Base case i=0: S[0...-1] empty. Mask for S[-K...-1].
    # Represents the state before filling S[0]. Conceptual history of K 'A's.
    # Mask 0 represents K 'A's (all bits 0). Count = 1.
    dp = [0] * (1 << K)
    dp[0] = 1

    for i in range(N): # We are filling index i
        new_dp = [0] * (1 << K)
        
        possible_chars = []
        if S[i] == 'A' or S[i] == '?':
            possible_chars.append('A')
        if S[i] == 'B' or S[i] == '?':
            possible_chars.append('B')

        for prev_mask in range(1 << K):
            if dp[prev_mask] == 0:
                continue

            for char_c in possible_chars:
                bit_c = 0 if char_c == 'A' else 1

                # Calculate the new mask for S[(i+1)-K ... (i+1)-1] = S[i-K+1 ... i]
                # prev_mask represents S[i-K ... i-1]
                # Bit j of prev_mask (0 to K-1) represents S[i-1-j]. LSB S[i-1], MSB S[i-K].
                # curr_mask represents S[i-K+1 ... i]
                # Bit j of curr_mask (0 to K-1) represents S[i-j]. LSB S[i], MSB S[i-K+1].
                # Bit 0 of curr_mask is S[i] which is bit_c.
                # Bit j (j>0) of curr_mask is S[i-j]. This is S[(i-1)-(j-1)], which is represented by bit j-1 of prev_mask.
                # curr_mask = bit_c (at LSB) | (prev_mask shifted left by 1)
                # curr_mask = (prev_mask << 1) | bit_c
                # Ensure mask stays within K bits by taking modulo 2^K or bitwise AND
                curr_mask = ((prev_mask << 1) | bit_c) & ((1 << K) - 1)

                # Check for palindrome S[i-K+1 ... i]. This substring is represented by curr_mask.
                # This check is relevant only if the substring is completely within the string S[0...N-1].
                # The start index i-K+1 must be >= 0, i.e., i >= K-1.
                is_pal = False
                if i >= K - 1: # The substring S[i-K+1...i] of length K is formed, starting at index i-K+1.
                    # Check if curr_mask represents a palindrome
                    # curr_mask = b_{K-1}...b_0 where b_j represents S[i-j].
                    # Palindrome if S[i-K+1+p] == S[i-p] for 0 <= p < K/2.
                    # S[i-p] is represented by bit p of curr_mask.
                    # S[i-K+1+p]. The index is i - (K-1-p). This is represented by bit K-1-p of curr_mask.
                    # Check bit p == bit K-1-p for 0 <= p < K/2.
                    temp_mask = curr_mask
                    is_pal = True
                    for p in range(K // 2):
                         # Check bit p vs bit K-1-p
                         if ((temp_mask >> p) & 1) != ((temp_mask >> (K-1-p)) & 1):
                             is_pal = False
                             break

                if not is_pal:
                    new_dp[curr_mask] = (new_dp[curr_mask] + dp[prev_mask]) % MOD

        dp = new_dp

    # The final answer is the sum of all counts in the last DP state (after processing S[N-1])
    ans = 0
    for mask in range(1 << K):
        ans = (ans + dp[mask]) % MOD

    print(ans)

solve()