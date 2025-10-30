def is_palindrome(window):
    return window == window[::-1]

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    N = int(data[0])
    K = int(data[1])
    S = data[2]

    MOD = 998244353

    # Pad the initial state with dummy values if K-1 > 0
    if K > 1:
        initial_state = tuple(S[:K-1]) if K-1 <= N else tuple()
    else:
        initial_state = tuple()

    # Initialize DP with empty state
    from collections import defaultdict
    prev_dp = defaultdict(int)
    prev_dp[initial_state] = 1

    for i in range(K-1, N):
        curr_dp = defaultdict(int)
        char = S[i]
        possible_chars = ['A', 'B'] if char == '?' else [char]
        for state in prev_dp:
            for c in possible_chars:
                new_state = tuple(list(state[1:]) + [c])
                window = tuple(list(state) + [c])
                if i >= K-1 and is_palindrome(window):
                    continue
                curr_dp[new_state] = (curr_dp[new_state] + prev_dp[state]) % MOD
        prev_dp = curr_dp

    total_count = sum(prev_dp.values()) % MOD
    print(total_count)

if __name__ == "__main__":
    main()