import sys

def main():
    input = sys.stdin.read().split()
    ptr = 0
    N = int(input[ptr])
    ptr += 1
    Q = int(input[ptr])
    ptr += 1
    
    adj = [set() for _ in range(N + 1)]
    isolated = N
    output = []
    
    for _ in range(Q):
        query_type = input[ptr]
        ptr += 1
        
        if query_type == '1':
            u = int(input[ptr])
            ptr += 1
            v = int(input[ptr])
            ptr += 1
            
            if len(adj[u]) == 0:
                isolated -= 1
            if len(adj[v]) == 0:
                isolated -= 1
            
            adj[u].add(v)
            adj[v].add(u)
        
        else:
            v = int(input[ptr])
            ptr += 1
            
            neighbors = list(adj[v])
            d = len(neighbors)
            
            for u in neighbors:
                adj[v].remove(u)
                adj[u].remove(v)
                if len(adj[u]) == 0:
                    isolated += 1
            
            if d > 0:
                isolated += 1
        
        output.append(str(isolated))
    
    print('
'.join(output))

if __name__ == "__main__":
    main()