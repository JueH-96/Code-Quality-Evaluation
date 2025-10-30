MOD = 998244353

def pow_mod(x, n):
    if n == 0:
        return 1
    elif n % 2 == 1:
        return x * pow_mod(x, n - 1) % MOD
    else:
        t = pow_mod(x, n // 2)
        return t * t % MOD

def solve(S):
    left = 0
    right = 0
    question = 0
    for s in S:
        if s == '(':
            left += 1
        elif s == ')':
            if left > 0:
                left -= 1
            elif question > 0:
                question -= 1
            else:
                right += 1
        else:
            question += 1

    if right > 0:
        return 0

    dp = [0] * (question + 1)
    dp[0] = 1
    for _ in range(left):
        for i in range(question, -1, -1):
            dp[i + 1] = (dp[i + 1] + dp[i]) % MOD
            dp[i] = (2 * dp[i]) % MOD

    return dp[question]

S = input()
print(solve(S))