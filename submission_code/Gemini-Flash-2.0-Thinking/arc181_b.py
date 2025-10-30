def solve():
    S = input()
    X = input()
    Y = input()

    def construct_f(s, t, z):
        res = ""
        for char in z:
            if char == '0':
                res += s
            else:
                res += t
        return res

    # Check if T = S is a solution
    fx_s = construct_f(S, S, X)
    fy_s = construct_f(S, S, Y)
    if fx_s == fy_s:
        print("Yes")
        return

    # Check if T = "" is a solution
    fx_empty = construct_f(S, "", X)
    fy_empty = construct_f(S, "", Y)
    if fx_empty == fy_empty:
        print("Yes")
        return

    print("No")

t = int(input())
for _ in range(t):
    solve()