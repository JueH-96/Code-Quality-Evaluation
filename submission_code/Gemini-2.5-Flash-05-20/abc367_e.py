import sys
from array import array

def solve():
    # Read N and K
    N, K = map(int, sys.stdin.readline().split())
    
    # Read X and A as lists of strings first for efficient parsing
    X_str = sys.stdin.readline().split()
    A_str = sys.stdin.readline().split()

    # Convert X to a 0-indexed array using array.array for memory efficiency.
    # 'I' type code stands for unsigned int, typically 4 bytes, sufficient for N up to 2*10^5.
    # X_i from input is 1-indexed, so we subtract 1.
    X_0idx = array('I', [int(x) - 1 for x in X_str])
    
    # Convert A to an array.array as well. Values are also within unsigned int range.
    A = array('I', [int(a) for a in A_str])

    # MAX_LOG_K is chosen to cover K up to 10^18.
    # log2(10^18) is approximately 59.79. So, 60 powers of 2 (from 2^0 to 2^59) are sufficient.
    MAX_LOG_K = 60 

    # dp[p][j] stores the index reached by starting at j and applying the operation 2^p times.
    # dp is a list of array.array objects for memory efficiency.
    dp = [array('I', [0]*N) for _ in range(MAX_LOG_K)]

    # Base case: dp[0][j] is the result after 2^0 = 1 operation.
    # This means moving from index j to X_0idx[j].
    for j in range(N):
        dp[0][j] = X_0idx[j]

    # Fill the dp table using binary lifting.
    # dp[p][j] is found by taking 2^(p-1) steps from j to an intermediate index,
    # and then another 2^(p-1) steps from that intermediate index.
    for p in range(1, MAX_LOG_K):
        for j in range(N):
            dp[p][j] = dp[p-1][dp[p-1][j]]

    # final_A will store the elements of the sequence after K operations.
    final_A = array('I', [0]*N)

    # For each position i in the final array A', determine which original element A[current_pos] maps to it.
    for i in range(N):
        current_pos = i # Start from the current index i.
        temp_K = K      # Use a temporary variable for K to iterate through its bits.

        # Iterate through the bits of K from LSB to MSB (0 up to MAX_LOG_K-1).
        # If the p-th bit of K is set, it means we need to take 2^p steps.
        # We use the precomputed dp[p][current_pos] to jump 2^p steps.
        for p in range(MAX_LOG_K):
            if (temp_K >> p) & 1: # Check if the p-th bit of temp_K is 1.
                current_pos = dp[p][current_pos] # Jump 2^p steps from current_pos.
        
        # After processing all bits of K, current_pos holds the original index
        # from which the value for A'[i] is taken.
        final_A[i] = A[current_pos]

    # Print the resulting sequence, space-separated.
    # Convert array.array to a list for easy printing with print(*...).
    print(*(list(final_A)))

# Call the main solve function.
solve()