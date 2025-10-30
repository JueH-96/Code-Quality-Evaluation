# YOUR CODE HERE
import sys
import array

def solve():
    MOD = 998244353
    import sys
    from sys import stdin
    input = sys.stdin.read
    data = input().split()
    N, Q = int(data[0]), int(data[1])
    ops = []
    V_set = set()
    idx = 2
    for _ in range(Q):
        P_i = int(data[idx])
        V_i = int(data[idx+1])
        ops.append((P_i, V_i))
        V_set.add(V_i)
        idx +=2
    sorted_V = sorted(V_set)
    sorted_V = [0] + sorted_V
    V_to_idx = {v:i for i, v in enumerate(sorted_V)}
    M = len(sorted_V) -1
    size = (M+1)*(M+2)//2
    dp_size = (M+1)*(M+2)//2
    dp = array.array('I', (0 for _ in range(dp_size)))
    def dp_idx(p, s):
        return p*(p+1)//2 + s
    dp[dp_idx(0,0)] =1
    for P_i, V_i in ops:
        V_idx = V_to_idx[V_i]
        dp_new = array.array('I', (0 for _ in range(dp_size)))
        for p in range(M+1):
            base = p*(p+1)//2
            for s in range(p+1):
                idx_current = base + s
                cnt = dp[idx_current]
                if cnt ==0:
                    continue
                # Assign to prefix
                if V_i >= sorted_V[p]:
                    new_p = V_idx
                    if sorted_V[new_p] >= sorted_V[s]:
                        idx_new = new_p*(new_p+1)//2 + s
                        dp_new[idx_new] = (dp_new[idx_new] + cnt) % MOD
                # Assign to suffix
                if V_i >= sorted_V[s]:
                    new_s = V_idx
                    if sorted_V[p] >= sorted_V[new_s]:
                        idx_new = p*(p+1)//2 + new_s
                        dp_new[idx_new] = (dp_new[idx_new] + cnt) % MOD
        dp = dp_new
    ans =0
    for p in range(M+1):
        base = p*(p+1)//2
        for s in range(p+1):
            ans = (ans + dp[base + s]) % MOD
    print(ans)