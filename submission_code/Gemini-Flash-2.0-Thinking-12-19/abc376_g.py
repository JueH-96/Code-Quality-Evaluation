def solve():
    n = int(input())
    parents = list(map(int, input().split()))
    a = list(map(int, input().split()))
    
    children = [[] for _ in range(n + 1)]
    for i in range(n):
        children[parents[i]].append(i + 1)
        
    subtree_a_sum = [0] * (n + 1)
    
    def calculate_subtree_sum(u):
        current_sum = a[u-1] if u > 0 else 0
        for v in children[u]:
            current_sum += calculate_subtree_sum(v)
        subtree_a_sum[u] = current_sum
        return current_sum
        
    for i in range(1, n + 1):
        calculate_subtree_sum(i)
        
    expected_ops_numerator = 0
    total_a_sum = sum(a)
    
    for treasure_location in range(1, n + 1):
        searched_vertices = {0}
        unsearched_vertices = set(range(1, n + 1))
        operations_count = 0
        
        while treasure_location not in searched_vertices:
            available_vertices = []
            for v in unsearched_vertices:
                if parents[v-1] in searched_vertices:
                    available_vertices.append(v)
                    
            if not available_vertices:
                break 
                
            best_vertex = -1
            max_subtree_sum = -1
            
            for v in available_vertices:
                current_subtree_sum = subtree_a_sum[v]
                if current_subtree_sum > max_subtree_sum:
                    max_subtree_sum = current_subtree_sum
                    best_vertex = v
                elif current_subtree_sum == max_subtree_sum:
                    if best_vertex == -1 or v < best_vertex:
                        best_vertex = v
                        
            searched_vertices.add(best_vertex)
            unsearched_vertices.remove(best_vertex)
            operations_count += 1
            
        expected_ops_numerator += a[treasure_location-1] * operations_count
        
    denominator = total_a_sum
    numerator = expected_ops_numerator
    
    mod = 998244353
    
    def power(base, exp):
        res = 1
        base %= mod
        while exp > 0:
            if exp % 2 == 1:
                res = (res * base) % mod
            exp >>= 1
            base = (base * base) % mod
        return res
        
    def mod_inverse(n):
        return power(n, mod - 2)
        
    if denominator == 0:
        result = 0
    else:
        inv_denominator = mod_inverse(denominator)
        result = (numerator * inv_denominator) % mod
        
    print(result)

t = int(input())
for _ in range(t):
    solve()