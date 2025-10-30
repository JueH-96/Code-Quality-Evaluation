def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    Q = int(data[1])
    
    degree = [0] * (N + 1)
    adj = [[] for _ in range(N + 1)]
    isolated_count = N
    
    index = 2
    outputs = []
    
    for _ in range(Q):
        query_type = data[index]
        if query_type == '1':
            u = int(data[index + 1])
            v = int(data[index + 2])
            index += 3
            if degree[u] == 0:
                isolated_count -= 1
            degree[u] += 1
            if degree[v] == 0:
                isolated_count -= 1
            degree[v] += 1
            adj[u].append(v)
            adj[v].append(u)
            outputs.append(str(isolated_count))
        elif query_type == '2':
            v = int(data[index + 1])
            index += 2
            for neighbor in adj[v]:
                degree[neighbor] -= 1
                if degree[neighbor] == 0:
                    isolated_count += 1
            degree[v] = 0
            adj[v].clear()
            if degree[v] == 0:
                isolated_count += 1
            outputs.append(str(isolated_count))
    
    print('
'.join(outputs))

if __name__ == "__main__":
    main()