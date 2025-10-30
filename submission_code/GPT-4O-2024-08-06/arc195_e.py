def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    MOD = 998244353
    
    # Read N and Q
    index = 0
    N = int(data[index])
    Q = int(data[index + 1])
    index += 2
    
    # Read A array
    A = [0] * (N + 1)
    for i in range(2, N + 1):
        A[i] = int(data[index])
        index += 1
    
    # Read queries
    queries = []
    for _ in range(Q):
        u = int(data[index])
        v = int(data[index + 1])
        queries.append((u, v))
        index += 2
    
    # Precompute factorials and inverse factorials
    factorial = [1] * (N + 1)
    for i in range(2, N + 1):
        factorial[i] = factorial[i - 1] * i % MOD
    
    # Precompute the sum of distances for each query
    results = []
    for u, v in queries:
        if u > v:
            u, v = v, u
        
        # Calculate the sum of distances for this query
        total_distance = 0
        for i in range(2, N + 1):
            if u < i <= v:
                total_distance += A[i] * factorial[i - 2] * factorial[N - i] % MOD
                total_distance %= MOD
        
        results.append(total_distance)
    
    # Output results
    for result in results:
        print(result)

main()