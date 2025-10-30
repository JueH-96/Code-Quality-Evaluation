def main():
    import sys
    data = sys.stdin.read().split()
    if not data:
        return
    s = data[0]
    n = len(s)
    mod = 998244353

    # dp[balance] represents the number of ways 
    # to form a valid prefix with a given balance.
    # Maximum balance can be at most n.
    dp = [0] * (n + 2)
    dp[0] = 1  # initial state: empty string, balance = 0

    for ch in s:
        new_dp = [0] * (n + 2)
        if ch == '(':
            # For '(' we always increase the balance.
            for bal in range(n + 1):
                if dp[bal]:
                    new_dp[bal + 1] = (new_dp[bal + 1] + dp[bal]) % mod
        elif ch == ')':
            # For ')' we decrease the balance, we can only do that if balance > 0.
            for bal in range(1, n + 2):
                if dp[bal]:
                    new_dp[bal - 1] = (new_dp[bal - 1] + dp[bal]) % mod
        else:  # ch == '?'
            # A '?' can be replaced by '(' or ')'
            # Option 1: replace with '('
            for bal in range(n + 1):
                if dp[bal]:
                    new_dp[bal + 1] = (new_dp[bal + 1] + dp[bal]) % mod
            # Option 2: replace with ')'
            for bal in range(1, n + 2):
                if dp[bal]:
                    new_dp[bal - 1] = (new_dp[bal - 1] + dp[bal]) % mod
        dp = new_dp

    # For the string to be a valid parenthesis string, the final balance must be zero.
    sys.stdout.write(str(dp[0] % mod))
    
if __name__ == '__main__':
    main()