def solve():
    n = int(input())
    s = str(n)
    v_n_str = s * n
    v_n = int(v_n_str)
    print(v_n % 998244353)

solve()