def main():
    import sys
    input_data = sys.stdin.read().strip().split()
    N = int(input_data[0])
    K = int(input_data[1])
    S = input_data[2]
    MOD = 998244353

    # If K > N, no need to handle because of constraints guaranteeing K <= N.
    # Precompute which K-bit masks represent a palindrome of length K.
    # We'll only use these if (i+1) >= K in our DP transitions.
    if K <= 10:
        # is_pal[mask] = True iff the K-bit pattern of mask is a palindrome
        is_pal = [False] * (1 << K)
        for mask in range(1 << K):
            # check palindrome
            # bit j from the right is (mask >> j) & 1
            # we compare for j in [0..K//2-1]
            pal = True
            for j in range(K // 2):
                left_bit = (mask >> j) & 1
                right_bit = (mask >> (K - 1 - j)) & 1
                if left_bit != right_bit:
                    pal = False
                    break
            is_pal[mask] = pal
    else:
        # The problem constraints guarantee K <= 10, so we should never get here.
        print(0)
        return

    # Helper to convert 'A'->0, 'B'->1. If '?' return all possibilities [0,1].
    def char_to_bits(c):
        if c == 'A':
            return [0]
        elif c == 'B':
            return [1]
        else:
            return [0, 1]  # '?'

    # We'll use a rolling DP approach:
    # dp array cur_dp[mask] = number of ways to form a prefix of length i
    # ending with the bit-pattern 'mask' (where we store up to K bits).
    cur_dp = [0] * (1 << K)
    cur_dp[0] = 1  # Empty prefix

    # Iterate over positions i in [0..N-1]
    for i in range(N):
        next_dp = [0] * (1 << K)
        possible_bits = char_to_bits(S[i])
        for mask in range(1 << K):
            ways = cur_dp[mask]
            if ways == 0:
                continue
            for b in possible_bits:
                # Shift left by 1, add b in the lowest bit, keep only K bits
                new_mask = ((mask << 1) & ((1 << K) - 1)) | b
                # Now we have prefix of length i+1
                # Check if we have a K-length substring to verify
                # i+1 >= K => i >= K-1
                if i + 1 >= K:  
                    # If these K bits form a palindrome, skip
                    if is_pal[new_mask]:
                        continue
                # Otherwise add to dp
                val = next_dp[new_mask] + ways
                if val >= MOD:
                    val -= MOD
                next_dp[new_mask] = val
        cur_dp = next_dp

    # The answer is sum over all possible masks after length N
    ans = sum(cur_dp) % MOD
    print(ans)

# Call main() at the end
if __name__ == "__main__":
    main()