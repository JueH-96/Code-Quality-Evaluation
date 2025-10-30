def solve():
    n = int(input())
    parents = list(map(int, input().split()))
    a = list(map(int, input().split()))
    
    total_a = sum(a)
    
    adj = [[] for _ in range(n + 1)]
    for i in range(n):
        adj[parents[i]].append(i + 1)
    
    expected_ops = 0
    
    for target_node in range(1, n + 1):
        prob = a[target_node - 1] / total_a
        
        ops = 0
        
        searched = [False] * (n + 1)
        searched[0] = True
        
        while not searched[target_node]:
            
            candidates = []
            for i in range(1, n + 1):
                if not searched[i] and searched[parents[i-1]]:
                    candidates.append(i)
            
            if not candidates:
                break
            
            
            best_node = -1
            min_expected_ops = float('inf')
            
            for node in candidates:
                temp_searched = searched[:]
                temp_searched[node] = True
                
                temp_expected_ops = 0
                
                for temp_target_node in range(1, n + 1):
                    if temp_target_node == target_node:
                        continue
                    
                    temp_ops = 0
                    temp_searched_copy = temp_searched[:]
                    
                    while not temp_searched_copy[temp_target_node]:
                        temp_candidates = []
                        for i in range(1, n + 1):
                            if not temp_searched_copy[i] and temp_searched_copy[parents[i-1]]:
                                temp_candidates.append(i)
                        
                        if not temp_candidates:
                            break
                        
                        temp_candidates.sort()
                        next_node = temp_candidates[0]
                        temp_searched_copy[next_node] = True
                        temp_ops += 1
                    
                    if temp_searched_copy[temp_target_node]:
                        temp_expected_ops += temp_ops * (a[temp_target_node - 1] / total_a)
                
                if temp_expected_ops < min_expected_ops:
                    min_expected_ops = temp_expected_ops
                    best_node = node
            
            searched[best_node] = True
            ops += 1
        
        expected_ops = (expected_ops + ops * prob) % 998244353
    
    print(int(expected_ops))

def mod_inv(a, m):
    m0 = m
    y = 0
    x = 1

    if (m == 1):
        return 0

    while (a > 1):
        q = a // m
        t = m

        m = a % m
        a = t
        t = y

        y = x - q * y
        x = t

    if (x < 0):
        x = x + m0

    return x

t = int(input())
for _ in range(t):
    solve()