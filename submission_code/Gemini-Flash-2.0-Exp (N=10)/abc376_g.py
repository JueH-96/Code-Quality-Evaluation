def solve():
    MOD = 998244353

    def mod_inv(x):
        return pow(x, MOD - 2, MOD)

    T = int(input())
    for _ in range(T):
        N = int(input())
        parents = list(map(int, input().split()))
        a = list(map(int, input().split()))

        total_a = sum(a)
        expected_ops = 0
        
        children = [[] for _ in range(N + 1)]
        for i in range(N):
            children[parents[i]].append(i + 1)
        
        for i in range(1, N + 1):
            curr_prob = a[i-1] * mod_inv(total_a) % MOD
            
            ops = 0
            q = [0]
            searched = [False] * (N + 1)
            searched[0] = True
            
            while not searched[i]:
                
                next_nodes = []
                for u in q:
                    for v in children[u]:
                        if not searched[v]:
                            next_nodes.append(v)
                
                if not next_nodes:
                    break
                
                best_next = -1
                best_prob = 0
                for next_node in next_nodes:
                    
                    prob_sum = 0
                    
                    q2 = [0]
                    searched2 = searched[:]
                    searched2[next_node] = True
                    
                    while not searched2[i]:
                        
                        next_nodes2 = []
                        for u2 in q2:
                            for v2 in children[u2]:
                                if not searched2[v2]:
                                    next_nodes2.append(v2)
                        
                        if not next_nodes2:
                            break
                        
                        
                        
                        max_prob = 0
                        best_next2 = -1
                        for next_node2 in next_nodes2:
                            if next_node2 == i:
                                max_prob = 1
                                best_next2 = next_node2
                                break
                            else:
                                max_prob_temp = a[next_node2-1] * mod_inv(total_a) % MOD
                                if max_prob_temp > max_prob:
                                    max_prob = max_prob_temp
                                    best_next2 = next_node2
                        
                        if best_next2 != -1:
                            searched2[best_next2] = True
                            q2 = [best_next2]
                        else:
                            break
                    
                    if searched2[i]:
                        prob_sum = 1
                    
                    if prob_sum > best_prob:
                        best_prob = prob_sum
                        best_next = next_node
                
                if best_next != -1:
                    searched[best_next] = True
                    q = [best_next]
                    ops += 1
                else:
                    break
            
            expected_ops = (expected_ops + (ops * curr_prob) % MOD) % MOD
        
        print(expected_ops)

solve()