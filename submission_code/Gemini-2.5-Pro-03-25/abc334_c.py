# YOUR CODE HERE
import sys

def solve():
    """
    Solves the problem of minimizing total weirdness when pairing socks.
    Reads input N, K, and the list A of colors with one missing sock.
    Calculates the minimum possible total weirdness using dynamic programming.
    Prints the result to stdout.
    """
    
    N, K = map(int, sys.stdin.readline().split())
    
    # If K=0, it means no socks were lost. We start with N pairs of socks (2 socks for each color 1 to N).
    # We have 2N socks in total and need to form floor(2N/2) = N pairs.
    # The optimal strategy is to form N pairs of type (i, i) for each color i from 1 to N.
    # Each such pair has weirdness |i-i| = 0.
    # The total minimum weirdness is 0.
    if K == 0:
        print(0)
        return

    # Read the K colors A_1, ..., A_K which have only one sock remaining after the loss.
    # The input guarantees these are distinct and sorted: A_1 < A_2 < ... < A_K.
    A = list(map(int, sys.stdin.readline().split()))
    
    # Overview of remaining socks:
    # - For each color `c` in the list `A`, we have 1 sock of color `c`. There are K such socks.
    # - For each color `c` from 1 to N not in `A`, we have 2 socks of color `c`. There are N-K such colors.
    
    # Total number of socks remaining is K * 1 + (N-K) * 2 = K + 2N - 2K = 2N - K.
    # We need to form P = floor((2N-K)/2) pairs.
    
    # Strategy:
    # Pairs formed from socks of the same color (i, i) have weirdness |i-i| = 0.
    # This is the minimum possible weirdness for any pair.
    # It's always optimal to form as many such pairs as possible first.
    # We can form N-K pairs of type (i, i) for the N-K colors not in A.
    # These pairs contribute 0 to the total weirdness.
    
    # After forming these N-K zero-weirdness pairs, we are left with K socks of colors A_1, ..., A_K.
    # We need to form P - (N-K) more pairs from these K socks.
    # Let's calculate how many pairs this is:
    # Number of additional pairs = floor((2N-K)/2) - (N-K).
    # Case 1: 2N-K is even. This means K must be even.
    #   Pairs needed = (2N-K)/2 - (N-K) = (2N-K - 2(N-K))/2 = (2N-K - 2N + 2K)/2 = K/2.
    #   We need to form K/2 pairs using the K socks. All K socks will be used.
    # Case 2: 2N-K is odd. This means K must be odd.
    #   Pairs needed = (2N-K-1)/2 - (N-K) = (2N-K-1 - 2(N-K))/2 = (2N-K-1 - 2N + 2K)/2 = (K-1)/2.
    #   We need to form (K-1)/2 pairs using the K socks. K-1 socks will be used, one sock will be left over.

    # The problem reduces to finding the minimum total weirdness by pairing the K socks A_1, ..., A_K according to these rules.
    
    # We use dynamic programming.
    # Let dp[i] be the minimum total weirdness using the first i socks from the sorted list A (i.e., socks with colors A_1, ..., A_i).
    # The definition of the DP state depends on the parity of i:
    # - If i is even, dp[i] is the minimum weirdness assuming all i socks are optimally paired.
    # - If i is odd, dp[i] is the minimum weirdness assuming i-1 socks are optimally paired and one sock among the first i is left over.
    
    # Initialize DP array of size K+1 with zeros.
    dp = [0] * (K + 1)
    
    # Base case: dp[0] = 0. With 0 socks, the minimum weirdness is 0.
    
    # Base case: dp[1]. We consider the first sock (color A_1). Since i=1 is odd, 
    # according to the DP state definition, one sock must be left over.
    # The only option is to leave A_1 unused. The minimum weirdness is 0.
    # This state is only relevant if K >= 1.
    if K >= 1:
      dp[1] = 0 
    
    # Compute dp[i] for i from 2 to K using the derived recurrence relations.
    for i in range(2, K + 1):
        # A is 0-indexed list in Python. A[i-1] corresponds to A_i (1-based index).
        # A[i-2] corresponds to A_{i-1} (1-based index).
        # The weirdness of pairing socks A_{i-1} and A_i is |A_i - A_{i-1}| = A[i-1] - A[i-2] since A is sorted.
        diff = A[i-1] - A[i-2]
        
        if i % 2 == 0:
            # i is even: All i socks must be paired.
            # The optimal pairing includes the pair (A_{i-1}, A_i) due to the property that pairing adjacent elements minimizes sum of differences.
            # The weirdness contribution of this pair is `diff`.
            # The remaining problem requires optimally pairing the first i-2 socks.
            # Since i-2 is even, all i-2 socks must be paired. The minimum cost for this is dp[i-2].
            # The recurrence relation is dp[i] = dp[i-2] + diff.
            dp[i] = dp[i-2] + diff
        else:
            # i is odd: One sock among the first i must be left unpaired.
            # Option 1: The i-th sock (color A_i, i.e., A[i-1]) is left unpaired.
            #   We must optimally pair the first i-1 socks. Since i-1 is even, all must be paired.
            #   The minimum cost for this subproblem is dp[i-1].
            # Option 2: The i-th sock (color A_i) is paired.
            #   For optimality, it must be paired with the (i-1)-th sock (color A_{i-1}, i.e., A[i-2]).
            #   The weirdness contribution of this pair (A_{i-1}, A_i) is `diff`.
            #   We must then optimally handle the first i-2 socks. Since i-2 is odd, one must be left unpaired.
            #   The minimum cost for this subproblem is dp[i-2].
            #   Total cost for Option 2 is dp[i-2] + diff.
            # We take the minimum cost between Option 1 and Option 2.
            dp[i] = min(dp[i-1], dp[i-2] + diff)

    # The final answer is dp[K]. This value represents the minimum total weirdness achievable
    # by pairing the K socks from list A according to the problem rules (all paired if K even, one left over if K odd).
    # Since the pairs formed from socks of colors not in A contribute 0 weirdness, dp[K] is the overall minimum total weirdness.
    print(dp[K])

# Execute the solve function to run the program.
solve()