import sys

def main():
    input = sys.stdin.read
    data = input().split()
    
    n = int(data[0])
    X = list(map(int, data[1:n+1]))
    P = list(map(int, data[n+1:2*n+1]))
    
    q = int(data[2*n+1])
    queries = list(zip(*[iter(map(int, data[2*n+2:]))]*2))
    
    # Create a list of (coordinate, population) tuples
    villages = sorted(zip(X, P))
    
    # Create a prefix sum array for the populations
    prefix_sum = [0]
    for _, p in villages:
        prefix_sum.append(prefix_sum[-1] + p)
    
    # Answer each query
    results = []
    for l, r in queries:
        left = bisect_left(villages, (l, -float('inf')))
        right = bisect_right(villages, (r, float('inf')))
        results.append(prefix_sum[right] - prefix_sum[left])
    
    # Output the results
    for result in results:
        print(result)

if __name__ == "__main__":
    import bisect
    main()