def solve():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    S = int(data[1])
    A = list(map(int, data[2:2+N]))
    
    # Use a hashmap to store the prefix sums and their indices
    prefix_sum = 0
    prefix_sums = {0: -1}  # To handle the case where the sum from the start matches S
    
    # We only need to consider the sequence up to 2N to find any possible subsequence
    # because any subsequence longer than N can be split into two parts that wrap around
    # and thus would be covered by two shorter subsequences within the first 2N elements.
    for i in range(2 * N):
        prefix_sum += A[i % N]
        
        # Check if there's a previous prefix sum such that current_sum - previous_sum = S
        if (prefix_sum - S) in prefix_sums:
            print("Yes")
            return
        
        # Store the current prefix sum if not already stored
        if prefix_sum not in prefix_sums:
            prefix_sums[prefix_sum] = i
    
    print("No")