def solve():
    n = int(input())
    mod = 998244353
    
    n_str = str(n)
    len_n = len(n_str)
    
    powers_of_10 = [1]
    for _ in range(19):
        powers_of_10.append((powers_of_10[-1] * 10) % mod)
    
    v_n = 0
    for _ in range(int(n)):
        v_n = (v_n * powers_of_10[len_n] + n) % mod
    
    print(v_n)

solve()