# YOUR CODE HERE
MOD = 998244353

def solve(S):
    n = len(S)
    dp = [0] * (n + 1)
    dp[0] = 1

    for i in range(1, n + 1):
        if S[i-1] == '?':
            dp[i] = (dp[i-1] * 52) % MOD
        else:
            dp[i] = dp[i-1]

        if i >= 4:
            count = 0
            for j in range(26):
                for k in range(26):
                    upper1 = chr(65 + j)
                    upper2 = chr(65 + k)
                    for lower in range(26):
                        lower = chr(97 + lower)
                        pattern = f"{upper1}{upper1}{lower}{upper2}"
                        match = True
                        p_idx = 0
                        for idx in range(i-4, i):
                            if S[idx] != '?' and S[idx] != pattern[p_idx]:
                                match = False
                                break
                            p_idx += 1
                        if match:
                            count += 1

            if count > 0:
                if S[i-1] == '?':
                    dp[i] = (dp[i] - count * dp[i-4]) % MOD
                else:
                    dp[i] = (dp[i] - dp[i-4]) % MOD

    return (dp[n] + MOD) % MOD

S = input().strip()
print(solve(S))