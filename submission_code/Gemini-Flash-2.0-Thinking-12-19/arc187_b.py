def solve():
    n, m = map(int, input().split())
    b = list(map(int, input().split()))
    minus_one_indices = [i for i, x in enumerate(b) if x == -1]
    num_minus_ones = len(minus_one_indices)
    total_components_sum = 0
    mod = 998244353
    
    import itertools
    
    replacements_choices = list(itertools.product(range(1, m + 1), repeat=num_minus_ones))
    
    for replacements in replacements_choices:
        current_b_prime = list(b)
        for i in range(num_minus_ones):
            current_b_prime[minus_one_indices[i]] = replacements[i]
        
        adjacency_list = [[] for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                if current_b_prime[i] <= current_b_prime[j]:
                    adjacency_list[i].append(j)
                    adjacency_list[j].append(i)
                    
        visited = [False] * n
        components_count = 0
        for i in range(n):
            if not visited[i]:
                components_count += 1
                queue = [i]
                visited[i] = True
                while queue:
                    u = queue.pop(0)
                    for v in adjacency_list[u]:
                        if not visited[v]:
                            visited[v] = True
                            queue.append(v)
                            
        total_components_sum = (total_components_sum + components_count) % mod
        
    print(total_components_sum)

if __name__ == '__main__':
    solve()