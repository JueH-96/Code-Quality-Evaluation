import sys

def main():
    input = sys.stdin.read().split()
    ptr = 0
    N = int(input[ptr])
    ptr += 1
    Q = int(input[ptr])
    ptr += 1
    
    adj = [set() for _ in range(N + 1)]
    degree = [0] * (N + 1)
    count_zero = N
    output = []
    
    for _ in range(Q):
        query_type = input[ptr]
        ptr += 1
        
        if query_type == '1':
            u = int(input[ptr])
            ptr += 1
            v = int(input[ptr])
            ptr += 1
            
            adj[u].add(v)
            adj[v].add(u)
            
            for node in (u, v):
                if degree[node] == 0:
                    count_zero -= 1
                degree[node] += 1
        
        else:
            v = int(input[ptr])
            ptr += 1
            
            neighbors = list(adj[v])
            for u in neighbors:
                adj[v].remove(u)
                adj[u].remove(v)
                
                degree[v] -= 1
                if degree[v] == 0:
                    count_zero += 1
                
                degree[u] -= 1
                if degree[u] == 0:
                    count_zero += 1
        
        output.append(str(count_zero))
    
    print('
'.join(output))

if __name__ == "__main__":
    main()