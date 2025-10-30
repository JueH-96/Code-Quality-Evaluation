import sys

def main():
    data = sys.stdin.read().split()
    index = 0
    N = int(data[index])
    index += 1
    Q = int(data[index])
    index += 1
    
    adj_list = [[] for _ in range(N + 1)]
    degree = [0] * (N + 1)
    isolated_count = N
    output = []
    
    for _ in range(Q):
        query_type = int(data[index])
        index += 1
        if query_type == 1:
            u = int(data[index])
            index += 1
            v = int(data[index])
            index += 1
            if degree[u] == 0:
                isolated_count -= 1
            if degree[v] == 0:
                isolated_count -= 1
            adj_list[u].append(v)
            adj_list[v].append(u)
            degree[u] += 1
            degree[v] += 1
            output.append(str(isolated_count))
        elif query_type == 2:
            v = int(data[index])
            index += 1
            neighbors = adj_list[v][:]
            adj_list[v] = []
            for neighbor in neighbors:
                degree[neighbor] -= 1
                if degree[neighbor] == 0:
                    isolated_count += 1
            degree[v] = 0
            if degree[v] == 0:
                isolated_count += 1
            output.append(str(isolated_count))
    sys.stdout.write('
'.join(output) + '
')

if __name__ == '__main__':
    main()