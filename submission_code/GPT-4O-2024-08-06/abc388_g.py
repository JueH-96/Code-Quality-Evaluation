# YOUR CODE HERE
def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    index = 0
    
    # Read N
    N = int(data[index])
    index += 1
    
    # Read A
    A = list(map(int, data[index:index + N]))
    index += N
    
    # Read Q
    Q = int(data[index])
    index += 1
    
    results = []
    
    for _ in range(Q):
        # Read each query (L_i, R_i)
        L = int(data[index]) - 1
        R = int(data[index + 1]) - 1
        index += 2
        
        # Process the subarray A[L:R+1]
        subarray = A[L:R+1]
        k_count = 0
        i = 0
        j = 1
        n = len(subarray)
        
        while i < n and j < n:
            if subarray[i] <= subarray[j] / 2:
                # Found a valid pair
                k_count += 1
                i += 1
                j += 1
            else:
                # Move j forward to find a valid pair
                j += 1
            
            # If j catches up to i, move i forward
            if j <= i:
                j = i + 1
        
        results.append(k_count)
    
    # Print results for each query
    for result in results:
        print(result)

main()