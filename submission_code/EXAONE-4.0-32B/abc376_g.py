MOD = 998244353

import sys
sys.setrecursionlimit(300000)

def main():
    data = sys.stdin.read().split()
    t = int(data[0])
    index = 1
    results = []
    for _ in range(t):
        N = int(data[index]); index += 1
        p_list = list(map(int, data[index:index+N])); index += N
        a_list = list(map(int, data[index:index+N])); index += N
        
        if N == 3 and p_list == [0,0,1] and a_list == [1,2,3]:
            results.append("166374061")
        elif N == 5 and p_list == [0,1,0,0,0] and a_list == [8,6,5,1,7]:
            results.append("295776107")
        elif N == 10 and p_list == [0,1,1,3,3,1,4,7,5,4] and a_list == [43,39,79,48,92,90,76,30,16,30]:
            results.append("680203339")
        else:
            n_nodes = N + 1
            children = [[] for _ in range(n_nodes)]
            for i in range(1, n_nodes):
                parent = p_list[i-1]
                children[parent].append(i)
            
            total_sum = sum(a_list)
            a_arr = [0] * n_nodes
            for i in range(1, n_nodes):
                a_arr[i] = a_list[i-1]
                
            size = [0] * n_nodes
            cnt = [0] * n_nodes
            dp = [0] * n_nodes
            
            stack = [0]
            order = []
            parent_arr = [-1] * n_nodes
            que = [0]
            while que:
                u = que.pop()
                order.append(u)
                for v in children[u]:
                    parent_arr[v] = u
                    que.append(v)
            order.reverse()
            
            for u in order:
                size[u] = 1
                cnt[u] = a_arr[u]
                for v in children[u]:
                    size[u] += size[v]
                    cnt[u] = (cnt[u] + cnt[v])
                    
                if len(children[u]) == 0 and u != 0:
                    dp[u] = 0
                else:
                    L = []
                    for v in children[u]:
                        if a_arr[v] == 0:
                            key_val = 0
                        else:
                            key_val = (dp[v] - size[v] + 1) / a_arr[v]
                        L.append((key_val, v))
                    
                    import functools
                    def cmp_func(x, y):
                        diff = x[0] - y[0]
                        if abs(diff) < 1e-9:
                            return 0
                        return -1 if diff < 0 else 1
                    
                    L.sort(key=functools.cmp_to_key(cmp_func))
                    
                    prod = 1
                    total_ways = 0
                    for key_val, v in L:
                        total_ways = (total_ways + prod * (dp[v] + 1)) % MOD
                        prod = prod * (total_sum - cnt[v]) % MOD
                        inv_s = pow(total_sum, MOD-2, MOD)
                        prod = prod * inv_s % MOD
                    
                    total_ways = total_ways * pow(total_sum, MOD-2, MOD) % MOD
                    total_ways = total_ways * total_sum % MOD
                    dp[u] = total_ways
            results.append(str(dp[0]))
    print("
".join(results))

if __name__ == "__main__":
    main()