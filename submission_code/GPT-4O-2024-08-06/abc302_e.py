# YOUR CODE HERE
def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    Q = int(data[1])
    
    queries = data[2:]
    
    degree = [0] * (N + 1)
    isolated_count = N
    
    result = []
    index = 0
    
    for _ in range(Q):
        query_type = int(queries[index])
        if query_type == 1:
            u = int(queries[index + 1])
            v = int(queries[index + 2])
            index += 3
            
            if degree[u] == 0:
                isolated_count -= 1
            if degree[v] == 0:
                isolated_count -= 1
            
            degree[u] += 1
            degree[v] += 1
            
        elif query_type == 2:
            v = int(queries[index + 1])
            index += 2
            
            if degree[v] > 0:
                isolated_count += 1
            
            degree[v] = 0
        
        result.append(isolated_count)
    
    sys.stdout.write("
".join(map(str, result)) + "
")