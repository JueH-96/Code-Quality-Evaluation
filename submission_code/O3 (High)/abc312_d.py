import sys

MOD = 998244353     # given modulus

def main() -> None:
    S = sys.stdin.readline().strip()
    n = len(S)

    # dp[bal] : number of ways for the current prefix whose balance is `bal`
    dp = [0] * (n + 2)          # +2 makes indexing a bit safer
    dp[0] = 1                   # empty prefix has balance 0

    for ch in S:
        nxt = [0] * (n + 2)

        if ch == '(':
            for bal in range(n):
                if dp[bal]:
                    nxt[bal + 1] = (nxt[bal + 1] + dp[bal]) % MOD

        elif ch == ')':
            for bal in range(1, n + 1):
                if dp[bal]:
                    nxt[bal - 1] = (nxt[bal - 1] + dp[bal]) % MOD

        else:   # ch == '?', try both '(' and ')'
            # treat as '('
            for bal in range(n):
                if dp[bal]:
                    nxt[bal + 1] = (nxt[bal + 1] + dp[bal]) % MOD
            # treat as ')'
            for bal in range(1, n + 1):
                if dp[bal]:
                    nxt[bal - 1] = (nxt[bal - 1] + dp[bal]) % MOD

        dp = nxt

    print(dp[0] % MOD)


if __name__ == "__main__":
    main()