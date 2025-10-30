# YOUR CODE HERE
def can_match_s(n, m, s, t):
    def can_replace(x, t, start):
        for i in range(m):
            if x[start + i] != '#' and x[start + i] != t[i]:
                return False
        return True

    def dfs(x, start):
        if start == n:
            return x == s
        if x == s:
            return True
        if s[start:start + m] == t:
            return dfs(x, start + 1)
        if can_replace(x, t, start):
            new_x = x[:start] + t + x[start + m:]
            if dfs(new_x, start + 1):
                return True
        return dfs(x, start + 1)

    x = '#' * n
    return "Yes" if dfs(x, 0) else "No"

import sys
input = sys.stdin.read
data = input().split()

n = int(data[0])
m = int(data[1])
s = data[2]
t = data[3]

print(can_match_s(n, m, s, t))