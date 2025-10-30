def solve():
    k = int(input())
    s = input()
    t = input()
    n = len(s)
    m = len(t)

    dp = {}
    def recurse(i, j, ops):
        if ops > k:
            return False
        if i == n and j == m:
            return ops <= k
        if (i, j, ops) in dp:
            return dp[(i, j, ops)]

        res = False
        if i < n and j < m and s[i] == t[j]:
            res = res or recurse(i + 1, j + 1, ops)
        
        if i < n:
            res = res or recurse(i + 1, j, ops + 1) #delete
        if j < m:
            res = res or recurse(i, j + 1, ops + 1) #insert
        if i < n and j < m:
            res = res or recurse(i + 1, j + 1, ops + 1) #replace

        dp[(i, j, ops)] = res
        return res

    if recurse(0, 0, 0):
        print("Yes")
    else:
        print("No")

solve()