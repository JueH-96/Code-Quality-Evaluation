import sys

def solve():
    """
    This function solves the problem using dynamic programming on a path.
    """
    N = int(sys.stdin.readline())
    s = sys.stdin.readline().strip()
    MOD = 998244353

    # f[i] will store the number of distinct in-degree sequences for the path graph 0..i.
    # g[i] will store the number of "symmetric" sequences among them.
    
    # Base case for the DP: a path with a single vertex 0.
    if s[0] == '0':
        f = 1
        g = 1
    else:  # s[0] == '1'
        f = 2
        g = 0

    # DP transition for i from 0 to N-2, building up the path.
    for i in range(N - 1):
        s_i = s[i]
        s_i_plus_1 = s[i + 1]

        f_prev, g_prev = f, g
        
        # The recurrence relations depend on the types of adjacent vertices.
        if s_i == '0' and s_i_plus_1 == '0':
            f = (2 * f_prev - g_prev + MOD) % MOD
            g = g_prev
        elif s_i == '0' and s_i_plus_1 == '1':
            f = (3 * f_prev - 2 * g_prev + MOD) % MOD
            g = f_prev
        elif s_i == '1' and s_i_plus_1 == '0':
            f = (2 * f_prev) % MOD
            g = g_prev
        elif s_i == '1' and s_i_plus_1 == '1':
            f = (3 * f_prev - g_prev + MOD) % MOD
            g = (f_prev - g_prev + MOD) % MOD

    # After the loop, f and g are for the path 0...N-1.
    # Now, we close the cycle by considering the edge {N-1, 0}.
    s_N_minus_1 = s[N-1]
    s_0 = s[0]
    
    f_final, g_final = f, g

    if s_N_minus_1 == '0' and s_0 == '0':
        ans = (2 * f_final - g_final + MOD) % MOD
    elif s_N_minus_1 == '0' and s_0 == '1':
        ans = (3 * f_final - 2 * g_final + MOD) % MOD
    elif s_N_minus_1 == '1' and s_0 == '0':
        ans = (2 * f_final) % MOD
    elif s_N_minus_1 == '1' and s_0 == '1':
        ans = (3 * f_final - g_final + MOD) % MOD
    
    print(ans)

solve()