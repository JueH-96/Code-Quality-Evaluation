memo = {}

def solve(n, a, x, y):
    if n == 0:
        return 0
    if n in memo:
        return memo[n]

    option1_cost = x + solve(n // a, a, x, y)

    expected_cost_dice = 0
    for b in range(1, 7):
        expected_cost_dice += solve(n // b, a, x, y)
    option2_cost = y + expected_cost_dice / 6.0

    memo[n] = min(option1_cost, option2_cost)
    return memo[n]

n, a, x, y = map(int, input().split())
result = solve(n, a, x, y)
print(f"{result:.10f}")