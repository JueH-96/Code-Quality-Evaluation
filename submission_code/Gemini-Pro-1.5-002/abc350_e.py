# YOUR CODE HERE
def solve():
    n, a, x, y = map(int, input().split())

    dp = {}

    def get_dp(curr_n):
        if curr_n == 0:
            return 0.0
        if curr_n in dp:
            return dp[curr_n]

        op1_cost = x + get_dp(curr_n // a)
        op2_cost = 0.0
        for b in range(1, 7):
            op2_cost += get_dp(curr_n // b)
        op2_cost = y + op2_cost / 6.0
        
        dp[curr_n] = min(op1_cost, op2_cost)
        return dp[curr_n]

    print(get_dp(n))

solve()