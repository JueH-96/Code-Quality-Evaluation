# YOUR CODE HERE
def process_queries(queries):
    A = []
    results = []
    
    for query in queries:
        parts = query.split()
        if parts[0] == '1':
            # Append operation
            x = int(parts[1])
            A.append(x)
        elif parts[0] == '2':
            # Find k-th from end operation
            k = int(parts[1])
            results.append(A[-k])
    
    return results

if __name__ == "__main__":
    import sys
    input = sys.stdin.read
    data = input().strip().split('
')
    
    Q = int(data[0])
    queries = data[1:Q+1]
    
    results = process_queries(queries)
    
    for result in results:
        print(result)