MOD = 998244353

def main():
    import sys
    input = sys.stdin.read().split()
    idx = 0
    N = int(input[idx]); idx += 1
    M = int(input[idx]); idx += 1
    K = int(input[idx]); idx += 1
    A = list(map(int, input[idx:idx + N]))
    idx += N

    # Precompute x^K mod MOD for all possible x
    max_xor = 0
    for a in A:
        max_xor |= a  # Find the maximum possible XOR value to optimize the size
    max_xor_bit = max_xor.bit_length()
    max_size = 1 << max_xor_bit

    # We need to track for each r mod M, the sum of x^K for each possible XOR x
    dp = [dict() for _ in range(M)]
    dp[0][0] = 0  # Initially, empty subsequence has XOR 0 and sum 0^K = 0

    for a in A:
        new_dp = [dict() for _ in range(M)]
        for r in range(M):
            for x in dp[r]:
                current_sum = dp[r][x]
                # Case 1: Do not take 'a', so r and x remain
                if x in new_dp[r]:
                    new_dp[r][x] = (new_dp[r][x] + current_sum) % MOD
                else:
                    new_dp[r][x] = current_sum % MOD
                # Case 2: Take 'a', new_r = (r + 1) % M, new_x = x ^ a
                new_r = (r + 1) % M
                new_x = x ^ a
                contribution = pow(new_x, K, MOD)
                if new_x in new_dp[new_r]:
                    new_dp[new_r][new_x] = (new_dp[new_r][new_x] + contribution) % MOD
                else:
                    new_dp[new_r][new_x] = contribution % MOD
        dp = new_dp

    # Sum all entries in dp[0], excluding the empty subsequence (x=0)
    total = 0
    for x in dp[0]:
        if x != 0:  # empty subsequence is not counted
            total = (total + dp[0][x]) % MOD
    print(total % MOD)

if __name__ == '__main__':
    main()