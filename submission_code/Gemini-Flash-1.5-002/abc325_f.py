# YOUR CODE HERE
N = int(input())
D = list(map(int, input().split()))
L1, C1, K1 = map(int, input().split())
L2, C2, K2 = map(int, input().split())

dp = {}
def solve(i, k1, k2):
    if i == N:
        return 0
    if (i, k1, k2) in dp:
        return dp[(i, k1, k2)]
    ans = float('inf')
    for num1 in range(min(k1 + 1, D[i] // L1 + 2)):
        for num2 in range(min(k2 + 1, (D[i] - num1 * L1) // L2 + 2)):
            if num1 * L1 + num2 * L2 >= D[i] and num1 <= k1 and num2 <= k2:
                ans = min(ans, num1 * C1 + num2 * C2 + solve(i + 1, k1 - num1, k2 - num2))
    dp[(i, k1, k2)] = ans
    return ans

res = solve(0, K1, K2)
if res == float('inf'):
    print(-1)
else:
    print(res)