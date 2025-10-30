# YOUR CODE HERE
import sys
import bisect

def main():
    """
    This function solves the problem by using dynamic programming on a reduced set of "key" vertices.
    """
    MOD = 998244353

    try:
        input = sys.stdin.readline
    except:
        pass

    # Read input
    N, M, K = map(int, input().split())
    
    if M == 0:
        print(1)
        return

    special_edges = [list(map(int, input().split())) for _ in range(M)]
    X = [edge[0] for edge in special_edges]
    Y = [edge[1] for edge in special_edges]

    # Set of key vertices: start vertex 1, and all start/end points of special edges
    key_vertices_set = {1}
    for x_i in X:
        key_vertices_set.add(x_i)
    for y_i in Y:
        key_vertices_set.add(y_i)
    
    key_vertices = sorted(list(key_vertices_set))
    q = len(key_vertices)
    v_map = {v: i for i, v in enumerate(key_vertices)}

    # Precomputation for the DP
    # prev_p_idx[i]: index of the key vertex one reaches by moving backwards from key_vertices[i]
    # dist_p[i]: number of default backward moves to reach prev_p[i]
    prev_p_idx = [0] * q
    dist_p = [0] * q
    for i in range(q):
        p = key_vertices[i]
        prev_p_val = key_vertices[i-1] 
        prev_p_idx[i] = (i-1 + q) % q
        dist_p[i] = (p - prev_p_val + N) % N
        if dist_p[i] == 0:
            dist_p[i] = N

    # falls_in[p_idx]: list of special edge indices {i} where Y[i] is on the default path from prev_p[p] to p
    falls_in = [[] for _ in range(q)]
    for i in range(M):
        y_i = Y[i]
        
        # Find the key vertex p such that y_i is in the cycle segment (prev_p, p]
        p_list_idx = bisect.bisect_left(key_vertices, y_i)
        if p_list_idx == q:
            p_list_idx = 0
        
        p_idx = p_list_idx
        p = key_vertices[p_idx]
        
        dist_y_to_p = (p - y_i + N) % N
        
        if dist_y_to_p < dist_p[p_idx]:
            falls_in[p_idx].append(i)

    # DP tables
    # dp[k][p_idx]: number of ways to reach vertex p in k steps
    # S[k][p_idx]: contribution to dp[k][p_idx] from paths with at least one special move
    dp = [[0] * q for _ in range(K + 1)]
    S = [[0] * q for _ in range(K + 1)]

    # Base case k=0: start at vertex 1
    if 1 in v_map:
        dp[0][v_map[1]] = 1
    
    # DP calculation
    for k in range(1, K + 1):
        dest_k = (k % N + 1) if k % N != 0 else N # Simplified from (1-1+k-1)%N + 1
        
        for p_idx in range(q):
            p = key_vertices[p_idx]
            
            # S[k][p] = S[k-d][p'] + correction
            d = dist_p[p_idx]
            p_prime_idx = prev_p_idx[p_idx]
            
            s_val = 0
            if k >= d:
                s_val = S[k - d][p_prime_idx]

            corr = 0
            for i in falls_in[p_idx]:
                # delta is the number of default moves from Y[i] to p
                delta = (p - Y[i] + N) % N
                if k - delta - 1 >= 0:
                    x_idx = v_map[X[i]]
                    corr = (corr + dp[k - delta - 1][x_idx]) % MOD
            
            S[k][p_idx] = (s_val + corr) % MOD

            # dp[k][p] = is_default_path_ending_here + S[k][p]
            is_def = 1 if p == dest_k else 0
            dp[k][p_idx] = (is_def + S[k][p_idx]) % MOD

    # Final calculation
    # Total ways = 1 (pure default path) + sum over all paths with at least one special move
    # The sum can be calculated as sum_{i=1 to M} sum_{k=0 to K-1} f[k][X_i]
    total_sum_special = 0
    for k in range(K):
        for i in range(M):
            x_idx = v_map[X[i]]
            total_sum_special = (total_sum_special + dp[k][x_idx]) % MOD
            
    ans = (1 + total_sum_special) % MOD
    
    print(ans)

if __name__ == "__main__":
    main()