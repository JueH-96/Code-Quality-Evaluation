def main():
    import sys
    input_data = sys.stdin.read().strip().split()
    N = int(input_data[0])
    K = int(input_data[1])
    S = input_data[2]

    MOD = 998244353

    # Precompute a helper to map 'A' -> 0, 'B' -> 1 for convenience
    # We'll treat '?' specially with a small check
    def char_to_bit(c):
        return 0 if c == 'A' else 1

    # Build a table to check which bit-patterns of length K are palindromes
    # We'll store palindromeCheck[x] = True/False
    # where x is a K-bit number. The leftmost bit is the most significant.
    palindromeCheck = [False] * (1 << K)  # for numbers from 0..(2^K - 1)

    for mask in range(1 << K):
        # Reconstruct the K bits
        # bits[0] will be the leftmost (most significant) bit
        # bits[K-1] will be the rightmost (least significant) bit
        bits = [(mask >> (K - 1 - i)) & 1 for i in range(K)]
        # Check if bits[0..K-1] is a palindrome
        is_pal = True
        for i in range(K // 2):
            if bits[i] != bits[K - 1 - i]:
                is_pal = False
                break
        palindromeCheck[mask] = is_pal

    # We will do a DP approach:
    # dp[pos][mask] = number of ways to form a partial string of length pos,
    # such that the last K bits (if pos >= K) or the pad+bits (if pos < K) is "mask".
    #
    # We'll implement space-efficient rolling:
    dp_current = [0] * (1 << K)
    dp_next = [0] * (1 << K)
    dp_current[0] = 1  # at pos=0, we have formed an empty string with mask=0

    for pos in range(N):
        # Clear dp_next
        for i in range(1 << K):
            dp_next[i] = 0

        c = S[pos]
        if c == '?':
            possible_next_bits = [0, 1]
        else:
            possible_next_bits = [char_to_bit(c)]

        for old_mask in range(1 << K):
            ways = dp_current[old_mask]
            if not ways:
                continue

            for nb in possible_next_bits:
                # new_mask: shift left by 1, add nb in the least significant bit,
                # and keep only the last K bits.
                new_mask = ((old_mask << 1) & ((1 << K) - 1)) | nb

                # If we already have at least K chars (pos+1 >= K),
                # check if the last K bits form a palindrome.
                # This is exactly new_mask, which is the last K bits
                if pos + 1 >= K and palindromeCheck[new_mask]:
                    continue  # skip, as we formed a palindrome of length K

                dp_next[new_mask] = (dp_next[new_mask] + ways) % MOD

        # Swap dp_current, dp_next
        dp_current, dp_next = dp_next, dp_current

    # After position N, any of the 2^K masks could be the final last K bits
    answer = sum(dp_current) % MOD
    print(answer)