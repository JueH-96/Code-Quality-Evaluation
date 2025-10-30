def main():
    import sys
    data = sys.stdin.read().split()
    if not data:
        return
    mod = 998244353
    N = int(data[0])
    K = int(data[1])
    S = data[2].strip()

    # dp[i] is a dictionary mapping the last min(i, K-1) characters (as a string)
    # of the constructed sequence to the number of ways (mod mod).
    dp = [{} for _ in range(N + 1)]
    dp[0][""] = 1

    # Function to check if a K-length string is a palindrome.
    def is_palindrome(s):
        for i in range(len(s) // 2):
            if s[i] != s[len(s) - 1 - i]:
                return False
        return True

    for i in range(N):
        cur_char = S[i]
        # For every state achieved at position i
        for state, ways in dp[i].items():
            # Determine possible letters at position i.
            possible = ['A', 'B'] if cur_char == '?' else [cur_char]
            for letter in possible:
                if len(state) == K - 1:
                    # Form a new substring of length K by appending the new letter.
                    candidate = state + letter
                    # If candidate is a palindrome, this is forbidden.
                    if is_palindrome(candidate):
                        continue
                    # Otherwise, remove the first character to maintain window of size K-1.
                    new_state = candidate[1:]
                else:
                    new_state = state + letter
                dp[i+1][new_state] = (dp[i+1].get(new_state, 0) + ways) % mod

    # The answer is the total count across all states of length N.
    result = sum(dp[N].values()) % mod
    sys.stdout.write(str(result))


if __name__ == '__main__':
    main()