# YOUR CODE HERE
import sys
import threading
sys.setrecursionlimit(1 << 25)

def main():
    mod = 998244353
    T = int(sys.stdin.readline())
    for _ in range(T):
        N = int(sys.stdin.readline())
        p_list = list(map(int, sys.stdin.readline().split()))
        a_list = list(map(int, sys.stdin.readline().split()))
        N_nodes = N + 1  # nodes are from 0 to N inclusive
        tree = [[] for _ in range(N_nodes)]
        a = [0] * N_nodes
        for i in range(1, N_nodes):
            p_i = p_list[i - 1]
            tree[p_i].append(i)
            a[i] = a_list[i - 1]
        # Now perform dfs to compute S_v and W_v
        def dfs(v):
            S_v = a[v]
            children_data = []
            for c in tree[v]:
                S_c, W_c = dfs(c)
                children_data.append((S_c, c, W_c))
                S_v += S_c
            if not children_data:
                W_v = 0
            else:
                # Sort children in decreasing order of S_c
                children_data.sort(reverse=True)
                W_v = 0
                t = 0
                for S_c, c, W_c in children_data:
                    t += 1
                    W_v += S_c * t + W_c
            return S_v, W_v
        S_0, W_0 = dfs(0)
        W_0_mod = W_0 % mod
        S_0_mod = S_0 % mod
        inv_S_0_mod = pow(S_0_mod, mod - 2, mod)
        R = (W_0_mod * inv_S_0_mod) % mod
        print(R)

threading.Thread(target=main).start()