def main():
    import sys
    data = sys.stdin.read().split()
    if not data:
        return
    mod = 998244353

    N = int(data[0])
    K = int(data[1])
    S = data[2].strip()

    # We use dynamic programming.
    # Let dp[i][state] be the number of ways to fill positions [0, i-1],
    # where "state" is the last min(i, K-1) letters of the string.
    # When len(state) == K-1 then adding a letter gives a block of K letters.
    # We only allow a transition if that new block (state + new letter) is not a palindrome.
    
    dp = [dict() for _ in range(N+1)]
    dp[0][""] = 1
    
    for i in range(N):
        current_char = S[i]
        for state, count in dp[i].items():
            if current_char == '?':
                choices = ['A','B']
            else:
                choices = [current_char]
                
            for c in choices:
                # If we already have K-1 characters in state then adding c
                # forms a substring of length K which must be checked.
                if len(state) == K-1:
                    block = state + c
                    if block == block[::-1]:
                        continue  # This block is a palindrome, so skip.
                    # New state: drop the first character and append c.
                    new_state = block[1:]
                else:
                    # If state length is less than K-1, simply append.
                    new_state = state + c
                dp[i+1][new_state] = (dp[i+1].get(new_state, 0) + count) % mod

    ans = sum(dp[N].values()) % mod
    sys.stdout.write(str(ans))

if __name__ == '__main__':
    main()