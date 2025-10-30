# YOUR CODE HERE
n = int(input())
a = list(map(int, input().split()))
s = []
t = []
for i in range(n - 1):
    si, ti = map(int, input().split())
    s.append(si)
    t.append(ti)

dp = {}
def solve(i, cur_a):
    if i == n - 1:
        return cur_a[i]
    if (i, tuple(cur_a)) in dp:
        return dp[(i, tuple(cur_a))]
    ans = solve(i + 1, cur_a)
    num_ops = cur_a[i] // s[i]
    for j in range(num_ops + 1):
        next_a = list(cur_a)
        next_a[i] -= j * s[i]
        next_a[i + 1] += j * t[i]
        ans = max(ans, solve(i + 1, next_a))
    dp[(i, tuple(cur_a))] = ans
    return ans

print(solve(0, a))