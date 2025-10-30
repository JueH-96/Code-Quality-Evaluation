# YOUR CODE HERE
n = int(input())
ans = (n + 2) // 5 * 5
if abs(ans - n) > abs((n + 2) // 5 * 5 - 5 - n):
    ans -= 5

print(ans)