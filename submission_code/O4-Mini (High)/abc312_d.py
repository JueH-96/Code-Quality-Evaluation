import sys
def main():
    S = sys.stdin.readline().strip()
    if not S:
        print(0)
        return
    n = len(S)
    MOD = 998244353
    # A well‐formed parentheses string must have even length.
    if n & 1:
        print(0)
        return

    # dp[j] = number of ways after processing i chars
    #         to have j unmatched '(' in the prefix
    dp = [0] * (n+1)
    dp[0] = 1

    for i, ch in enumerate(S):
        # after i chars, dp[j] is only possibly nonzero for j <= i
        nxt = [0] * (n+1)
        # if we can place '(' here
        if ch != ')':
            # transition: j -> j+1
            # j runs 0..i
            for j in range(i+1):
                v = dp[j]
                if v:
                    nxt[j+1] = v
        # if we can place ')' here
        if ch != '(':
            # transition: j -> j-1  (only if j>=1)
            # j runs 1..i+1 but dp[j] is nonzero only up to j=i
            for j in range(1, i+2):
                v = dp[j]
                if v:
                    s = nxt[j-1] + v
                    if s >= MOD:
                        s -= MOD
                    nxt[j-1] = s
        dp = nxt

    # answer is the number of ways to end with 0 unmatched '('
    print(dp[0] % MOD)

if __name__ == "__main__":
    main()