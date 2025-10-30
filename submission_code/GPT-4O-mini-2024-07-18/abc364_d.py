def main():
    import sys
    input = sys.stdin.read
    data = input().splitlines()
    
    # Read N and Q
    N, Q = map(int, data[0].split())
    
    # Read points A
    A = list(map(int, data[1].split()))
    
    results = []
    
    # Process each query
    for i in range(2, 2 + Q):
        b_j, k_j = map(int, data[i].split())
        
        # Calculate distances from b_j to each point in A
        distances = [abs(a_i - b_j) for a_i in A]
        
        # Sort distances
        distances.sort()
        
        # Get the k_j-th closest distance (1-indexed)
        results.append(distances[k_j - 1])
    
    # Print results
    print('
'.join(map(str, results)))

if __name__ == "__main__":
    main()