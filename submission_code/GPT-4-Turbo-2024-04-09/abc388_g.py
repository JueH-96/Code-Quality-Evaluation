def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    index = 0
    
    # Read number of mochi
    N = int(data[index])
    index += 1
    
    # Read sizes of mochi
    A = list(map(int, data[index:index+N]))
    index += N
    
    # Read number of queries
    Q = int(data[index])
    index += 1
    
    queries = []
    for _ in range(Q):
        L = int(data[index]) - 1  # Convert to zero-based index
        R = int(data[index+1]) - 1  # Convert to zero-based index
        queries.append((L, R))
        index += 2
    
    # Function to handle each query
    def handle_query(L, R):
        # Extract the subarray of mochi sizes for this query
        sub_mochi = A[L:R+1]
        n = len(sub_mochi)
        
        # Two pointers technique to find maximum kagamimochi
        i = 0
        j = 0
        count = 0
        
        while i < n and j < n:
            # Move j to find a mochi that is at least twice as large as A[i]
            while j < n and sub_mochi[j] < 2 * sub_mochi[i]:
                j += 1
            if j < n:
                # We can form one kagamimochi
                count += 1
                # Move both pointers to consider next possible pair
                i += 1
                j += 1
        
        return count
    
    results = []
    for L, R in queries:
        results.append(handle_query(L, R))
    
    # Print all results
    for result in results:
        print(result)

if __name__ == "__main__":
    main()