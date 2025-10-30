def next_garbage_collection(N, garbage_types, Q, queries):
    results = []
    
    for t_j, d_j in queries:
        q, r = garbage_types[t_j - 1]
        # Calculate the next collection day
        if d_j % q == r:
            results.append(d_j)
        else:
            # Find the next day when d_j % q == r
            next_day = d_j + (r - d_j % q + q) % q
            results.append(next_day)
    
    return results

def main():
    import sys
    input = sys.stdin.read
    data = input().splitlines()
    
    N = int(data[0])
    garbage_types = []
    
    for i in range(1, N + 1):
        q, r = map(int, data[i].split())
        garbage_types.append((q, r))
    
    Q = int(data[N + 1])
    queries = []
    
    for i in range(N + 2, N + 2 + Q):
        t_j, d_j = map(int, data[i].split())
        queries.append((t_j, d_j))
    
    results = next_garbage_collection(N, garbage_types, Q, queries)
    
    for result in results:
        print(result)

if __name__ == "__main__":
    main()