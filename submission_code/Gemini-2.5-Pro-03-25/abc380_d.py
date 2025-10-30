# YOUR CODE HERE
import sys

# Setting a higher recursion depth limit just in case, although T_eff=60 should be fine.
# The default Python recursion depth limit is typically 1000.
try:
    # Increase recursion depth limit. This is a safety measure for deep recursive calls.
    # For T_eff = 60, this might not be strictly necessary but is good practice in competitive programming.
    sys.setrecursionlimit(2000) 
except Exception as e:
    # Some environments might restrict changing recursion depth (e.g., online judges).
    # If setting fails, we proceed with the default limit, hoping it's sufficient.
    # print(f"Warning: Could not set recursion depth: {e}", file=sys.stderr) # Optional warning
    pass


def solve():
    # Read the initial string S from standard input
    S = sys.stdin.readline().strip()
    L0 = len(S) # Store the length of the initial string

    # Read the number of queries Q
    Q = int(sys.stdin.readline())
    
    # Handle the edge case where there are no queries.
    if Q == 0:
        # Print an empty line (or follow specific contest rules for empty output).
        print() 
        return
        
    # Read the query indices K_i (1-based)
    Ks = list(map(int, sys.stdin.readline().split()))

    # List to store the resulting characters for each query
    results = []
    
    # Memoization cache dictionary: stores results for (t, k) pairs to avoid recomputation.
    # Key: tuple (t, k) representing state (time level, index)
    # Value: tuple (final_k, C) result from trace(t, k)
    memo = {}

    # Recursive function to trace the character at index k in string S_t back to its origin in S_0.
    # Parameters:
    # t: The effective number of operations considered (time level). Represents S_t.
    # k: 1-based index of the character we are interested in within the conceptual string S_t.
    # Returns: A tuple (final_k, C) where:
    #          final_k is the 1-based index in the original string S_0 from which the character originates.
    #          C is the total number of case swaps applied to the character along the path from S_0.
    def trace(t, k):
        
        # Check if the result for state (t, k) is already computed and stored in the cache.
        state = (t, k)
        if state in memo:
            return memo[state]

        # Base case: If t=0, we have reached the initial string S_0.
        if t == 0:
            # The character's origin is at index k in S_0. No swaps induced at level 0.
            return (k, 0) # Return (1-based index in S_0, swap count = 0)

        # Optimization: The structure of the string for indices up to 10^18 stabilizes after roughly 60 steps.
        # If t >= 61, the length of S_{t-1} (L0 * 2^(t-1)) is guaranteed to be greater than 10^18.
        # Since any query index k is at most 10^18, k must fall within the first half (S_{t-1}).
        if t >= 61:
             # The character comes from S_{t-1} at the same index k. Recursively call trace for t-1.
             # The swap count does not change in this step.
             res = trace(t - 1, k)
             # Cache the result for this state before returning.
             memo[state] = res
             return res

        # For t < 61, we need to calculate the length of S_{t-1} which acts as the boundary.
        # Python's arbitrary precision integers handle potentially large values.
        # boundary = L0 * 2**(t-1)
        boundary = L0 * (1 << (t - 1))

        # Check if index k falls in the first half (a copy of S_{t-1}) or the second half (T_{t-1}).
        if k <= boundary:
            # k is in the first half. The character originates from S_{t-1} at the same index k.
            # Recursively find the origin in S_{t-1}. Swap count remains unchanged from the recursive call.
            res = trace(t - 1, k)
        else:
            # k is in the second half (T_{t-1}). T_{t-1} is the case-swapped version of S_{t-1}.
            # The corresponding index within S_{t-1} is k - boundary.
            # Recursively find the origin using the adjusted index.
            # The total swap count increases by 1 because this step involves a case swap.
            final_k, C = trace(t - 1, k - boundary)
            res = (final_k, C + 1)
        
        # Store the computed result in the memoization cache before returning.
        memo[state] = res
        return res

    # Process each query K_i provided in the input
    for K_i in Ks:
        # The problem involves 10^100 operations, but for indices up to 10^18,
        # the character depends only on the first ~60 operations.
        # We use an effective number of operations T_eff = 60 for the trace function.
        T_eff = 60 
        
        # Call the recursive function 'trace' starting from level T_eff and index K_i.
        # This finds the original index in S_0 (final_k) and the total swap count (C).
        final_k, C = trace(T_eff, K_i)
        
        # final_k is the 1-based index into the original string S.
        # Convert it to a 0-based index for Python string access.
        base_char = S[final_k - 1]

        # Determine the final character based on the parity of the swap count C.
        # If C is odd, the final character is the case-swapped version of the base character.
        if C % 2 == 1:
            results.append(base_char.swapcase())
        else: # If C is even, the final character has the same case as the base character.
            results.append(base_char)

    # Print the collected results for all queries, separated by spaces, followed by a newline.
    print(*(results))

# Execute the main function to solve the problem based on standard input.
solve()