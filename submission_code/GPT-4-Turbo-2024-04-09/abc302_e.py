import sys
input = sys.stdin.read

def main():
    data = input().split()
    index = 0
    N = int(data[index])
    index += 1
    Q = int(data[index])
    index += 1
    
    # Adjacency list to keep track of connections
    connections = [set() for _ in range(N + 1)]
    # Number of isolated vertices
    isolated_count = N
    
    results = []
    
    for _ in range(Q):
        query_type = int(data[index])
        index += 1
        
        if query_type == 1:
            u = int(data[index])
            index += 1
            v = int(data[index])
            index += 1
            
            # Connect u and v
            if len(connections[u]) == 0:
                isolated_count -= 1
            if len(connections[v]) == 0:
                isolated_count -= 1
            
            connections[u].add(v)
            connections[v].add(u)
        
        elif query_type == 2:
            v = int(data[index])
            index += 1
            
            # Disconnect v from all its connections
            if len(connections[v]) > 0:
                for u in connections[v]:
                    connections[u].remove(v)
                connections[v].clear()
                isolated_count += 1
        
        # Output the number of isolated vertices
        results.append(str(isolated_count))
    
    print("
".join(results))

main()