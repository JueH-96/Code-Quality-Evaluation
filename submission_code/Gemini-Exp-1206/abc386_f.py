def solve():
    k = int(input())
    s = input()
    t = input()
    n = len(s)
    m = len(t)
    
    dp = {}

    def get_dp(i, j):
        if (i, j) in dp:
            return dp[(i, j)]

        if i == n:
            dp[(i, j)] = m - j
            return dp[(i, j)]
        
        if j == m:
            dp[(i, j)] = n - i
            return dp[(i, j)]

        if s[i] == t[j]:
            dp[(i, j)] = get_dp(i + 1, j + 1)
        else:
            dp[(i, j)] = 1 + min(get_dp(i + 1, j), get_dp(i, j + 1), get_dp(i + 1, j + 1))
        
        return dp[(i, j)]

    if get_dp(0, 0) <= k:
        print("Yes")
    else:
        print("No")

solve()