# YOUR CODE HERE
import sys

def main():
    """
    This function implements the dynamic programming solution described above.
    """
    N, M = map(int, sys.stdin.readline().split())
    S = sys.stdin.readline().strip()
    MOD = 998244353

    # dp[mask] represents the number of strings of the current length
    # that result in the LCS DP row profile represented by `mask`.
    # A mask's i-th bit being 1 means the LCS length increases at S[i].
    dp = [0] * (1 << N)
    dp[0] = 1

    memo_transitions = {}

    def get_next_mask(mask, char):
        """
        Calculates the next state mask given a current mask and a new character.
        Memoization is used to speed up repeated calculations.
        """
        state_key = (mask, char)
        if state_key in memo_transitions:
            return memo_transitions[state_key]

        # Convert mask to the DP row representation.
        # current_row[i] is the LCS length with S[:i].
        current_row = [0] * (N + 1)
        for i in range(N):
            # The i-th bit of the mask corresponds to the difference
            # between current_row[i+1] and current_row[i].
            current_row[i + 1] = current_row[i] + ((mask >> i) & 1)

        # Compute the next DP row based on the standard LCS recurrence.
        next_row = [0] * (N + 1)
        for i in range(1, N + 1):
            if S[i - 1] == char:
                next_row[i] = current_row[i - 1] + 1
            else:
                next_row[i] = max(next_row[i - 1], current_row[i])
        
        # Convert the newly computed DP row back to its mask representation.
        next_mask = 0
        for i in range(N):
            if next_row[i + 1] > next_row[i]:
                next_mask |= (1 << i)

        memo_transitions[state_key] = next_mask
        return next_mask

    # Group characters for optimized transitions.
    # Characters in S are handled individually.
    # All other characters behave identically.
    unique_chars_S = sorted(list(set(S)))
    other_char = ''
    s_chars_set = set(S)
    # Find a character that is not in S. Since N<=10, such a character always exists.
    for c_code in range(ord('a'), ord('z') + 1):
        c = chr(c_code)
        if c not in s_chars_set:
            other_char = c
            break
    other_chars_count = 26 - len(unique_chars_S)

    # Main DP loop, building the string T of length M character by character.
    for _ in range(M):
        new_dp = [0] * (1 << N)
        for mask in range(1 << N):
            if dp[mask] == 0:
                continue
            count = dp[mask]

            # Transition for each unique character present in S.
            for char in unique_chars_S:
                next_mask = get_next_mask(mask, char)
                new_dp[next_mask] = (new_dp[next_mask] + count) % MOD
            
            # Transition for all characters not in S, grouped together.
            if other_chars_count > 0:
                next_mask_other = get_next_mask(mask, other_char)
                new_dp[next_mask_other] = (new_dp[next_mask_other] + count * other_chars_count) % MOD
        dp = new_dp
    
    # Aggregate results. The final LCS length is the number of set bits in the mask.
    ans = [0] * (N + 1)
    for mask in range(1 << N):
        if dp[mask] > 0:
            lcs_len = bin(mask).count('1')
            ans[lcs_len] = (ans[lcs_len] + dp[mask]) % MOD
    
    print(*ans)

if __name__ == "__main__":
    main()