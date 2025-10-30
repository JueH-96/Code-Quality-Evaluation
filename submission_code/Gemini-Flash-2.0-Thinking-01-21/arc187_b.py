def solve():
    n, m = map(int, input().split())
    b = list(map(int, input().split()))
    
    minus_one_indices = []
    for i in range(n):
        if b[i] == -1:
            minus_one_indices.append(i)
    
    num_minus_ones = len(minus_one_indices)
    total_sequences = m ** num_minus_ones
    total_components_sum = 0
    
    if num_minus_ones == 0:
        sequence = b
        adj = [[] for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                if sequence[i] <= sequence[j]:
                    adj[i].append(j)
                    adj[j].append(i)
        
        visited = [False] * n
        components = 0
        for i in range(n):
            if not visited[i]:
                components += 1
                q = [i]
                visited[i] = True
                while q:
                    u = q.pop(0)
                    for v in adj[u]:
                        if not visited[v]:
                            visited[v] = True
                            q.append(v)
        print(components)
        return
        
    sequences = []
    
    def generate_sequences(current_sequence, index):
        if index == num_minus_ones:
            sequences.append(list(current_sequence))
            return
            
        current_index_to_fill = minus_one_indices[index]
        for value in range(1, m + 1):
            current_sequence[current_index_to_fill] = value
            generate_sequences(current_sequence, index + 1)
            
    initial_sequence = list(b)
    generate_sequences(initial_sequence, 0)
    
    for sequence in sequences:
        adj = [[] for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                if sequence[i] <= sequence[j]:
                    adj[i].append(j)
                    adj[j].append(i)
                    
        visited = [False] * n
        components = 0
        for i in range(n):
            if not visited[i]:
                components += 1
                q = [i]
                visited[i] = True
                while q:
                    u = q.pop(0)
                    for v in adj[u]:
                        if not visited[v]:
                            visited[v] = True
                            q.append(v)
        total_components_sum += components
        
    print(total_components_sum % 998244353)

if __name__ == '__main__':
    solve()