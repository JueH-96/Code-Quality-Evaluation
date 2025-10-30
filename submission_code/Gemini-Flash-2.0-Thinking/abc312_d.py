def solve():
    s = input()
    n = len(s)
    mod = 998244353

    dp = {0: 1}

    for char in s:
        new_dp = {}
        for balance, count in dp.items():
            if char == '(':
                new_balance = balance + 1
                new_dp[new_balance] = (new_dp.get(new_balance, 0) + count) % mod
            elif char == ')':
                if balance > 0:
                    new_balance = balance - 1
                    new_dp[new_balance] = (new_dp.get(new_balance, 0) + count) % mod
            elif char == '?':
                # Replace with '('
                new_balance_open = balance + 1
                new_dp[new_balance_open] = (new_dp.get(new_balance_open, 0) + count) % mod
                # Replace with ')'
                if balance > 0:
                    new_balance_close = balance - 1
                    new_dp[new_balance_close] = (new_dp.get(new_balance_close, 0) + count) % mod
        dp = new_dp

    print(dp.get(0, 0))

solve()