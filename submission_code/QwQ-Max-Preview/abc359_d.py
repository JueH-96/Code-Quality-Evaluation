mod = 998244353

def main():
    import sys
    from collections import defaultdict

    N, K = map(int, sys.stdin.readline().split())
    S = sys.stdin.readline().strip()

    mask_length = K - 1
    mask = (1 << mask_length) - 1

    dp = [defaultdict(int) for _ in range(N + 2)]  # dp[i] is after processing i-th character (0-based)

    # Generate initial states for first mask_length characters
    initial_states = []
    for s in range(0, 1 << mask_length):
        valid = True
        for pos_in_mask in range(mask_length):
            string_pos = pos_in_mask
            if string_pos >= N:
                valid = False
                break
            c = S[string_pos]
            bit = (s >> (mask_length - 1 - pos_in_mask)) & 1
            if c == 'A' and bit != 0:
                valid = False
                break
            if c == 'B' and bit != 1:
                valid = False
                break
        if valid:
            initial_states.append(s)
    
    for s in initial_states:
        dp[mask_length][s] = 1

    # Process each position from mask_length (K-1) to N-1
    for i in range(mask_length, N):
        current_char = S[i]
        possible_chars = []
        if current_char == 'A':
            possible_chars = [0]
        elif current_char == 'B':
            possible_chars = [1]
        elif current_char == '?':
            possible_chars = [0, 1]
        else:
            assert False, "Invalid character"

        for state in list(dp[i].keys()):
            count = dp[i][state]
            for next_char in possible_chars:
                is_palin = True
                for step in range(K // 2):
                    left = step
                    right = K - 1 - step
                    # Get bits for left and right positions in the substring (state + next_char)
                    if left < mask_length:
                        bit_left = (state >> (mask_length - 1 - left)) & 1
                    else:
                        bit_left = next_char
                    if right < mask_length:
                        bit_right = (state >> (mask_length - 1 - right)) & 1
                    else:
                        bit_right = next_char
                    if bit_left != bit_right:
                        is_palin = False
                        break
                if is_palin:
                    continue
                new_state = (state << 1 | next_char) & mask
                dp[i+1][new_state] = (dp[i+1][new_state] + count) % mod

    # Sum all possible states in the final position
    print(sum(dp[N].values()) % mod)

if __name__ == "__main__":
    main()