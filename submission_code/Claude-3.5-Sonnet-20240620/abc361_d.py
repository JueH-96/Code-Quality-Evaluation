# YOUR CODE HERE
def min_operations(N, S, T):
    def count_mismatches(s1, s2):
        return sum(a != b for a, b in zip(s1, s2))

    def can_achieve(s1, s2):
        return sorted(s1) == sorted(s2)

    if not can_achieve(S, T):
        return -1

    dp = {}

    def solve(s, t):
        if s == t:
            return 0
        if len(s) == 2:
            return 1 if s != t else 0

        key = (s, t)
        if key in dp:
            return dp[key]

        result = float('inf')
        for i in range(len(s) - 1):
            new_s = s[:i] + s[i+2:] + s[i:i+2]
            result = min(result, 1 + solve(new_s, t))

        dp[key] = result
        return result

    return solve(S, T)

N = int(input())
S = input().strip()
T = input().strip()

print(min_operations(N, S, T))