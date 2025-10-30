# YOUR CODE HERE
def process_queries(N, A, B, queries):
    results = []
    
    for query in queries:
        parts = query.split()
        query_type = int(parts[0])
        
        if query_type == 1:
            # Type 1 query: 1 i x
            i = int(parts[1]) - 1
            x = int(parts[2])
            A[i] = x
        
        elif query_type == 2:
            # Type 2 query: 2 i x
            i = int(parts[1]) - 1
            x = int(parts[2])
            B[i] = x
        
        elif query_type == 3:
            # Type 3 query: 3 l r
            l = int(parts[1]) - 1
            r = int(parts[2]) - 1
            
            v = 0
            for i in range(l, r + 1):
                # Choose the operation that maximizes v
                if v + A[i] > v * B[i]:
                    v += A[i]
                else:
                    v *= B[i]
            
            results.append(v)
    
    return results

def main():
    import sys
    input = sys.stdin.read
    data = input().splitlines()
    
    N = int(data[0])
    A = list(map(int, data[1].split()))
    B = list(map(int, data[2].split()))
    Q = int(data[3])
    queries = data[4:4+Q]
    
    results = process_queries(N, A, B, queries)
    
    for result in results:
        print(result)

if __name__ == "__main__":
    main()