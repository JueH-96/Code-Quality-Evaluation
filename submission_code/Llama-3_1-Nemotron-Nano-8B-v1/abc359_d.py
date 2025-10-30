MOD = 998244353

def is_palindrome(s):
    return s == s[::-1]

def main():
    import sys
    from collections import defaultdict

    N, K = map(int, sys.stdin.readline().split())
    S = sys.stdin.readline().strip()

    current_dp = defaultdict(int)
    current_dp[""] = 1  # Initial state: no characters processed

    for i in range(N):
        next_dp = defaultdict(int)
        current_char = S[i]
        possible_chars = ['A', 'B'] if current_char == '?' else [current_char]
        for s, cnt in current_dp.items():
            for c in possible_chars:
                # Check if adding this character would form a palindrome window
                if i >= K - 1:
                    window = s + c
                    if is_palindrome(window):
                        continue  # Skip this transition
                # Update the state
                if len(s) < K - 1:
                    new_s = s + c
                else:
                    new_s = s[1:] + c
                next_dp[new_s] = (next_dp[new_s] + cnt) % MOD
        current_dp = next_dp

    # Sum all possible states after processing all characters
    print(sum(current_dp.values()) % MOD)

if __name__ == "__main__":
    main()