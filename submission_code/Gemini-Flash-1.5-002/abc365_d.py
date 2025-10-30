# YOUR CODE HERE
n = int(input())
s = input()
win = {'R': 'P', 'S': 'R', 'P': 'S'}
lose = {'R': 'S', 'S': 'P', 'P': 'R'}
dp = {}
def solve(i, last, won):
    if i == n:
        return won
    if (i, last, won) in dp:
        return dp[(i, last, won)]
    ans = solve(i + 1, s[i], won)
    if s[i] != last and win[s[i]] != last:
        ans = max(ans, solve(i + 1, win[s[i]], won + 1))
    dp[(i, last, won)] = ans
    return ans
print(solve(0, '', 0))