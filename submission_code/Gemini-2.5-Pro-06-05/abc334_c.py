import sys

def solve():
    """
    Reads input, solves the socks pairing problem, and prints the result.
    """
    # Read problem parameters
    try:
        line1 = sys.stdin.readline()
        if not line1: return
        N, K = map(int, line1.split())
    except (IOError, ValueError):
        return
        
    # According to constraints 1 <= K <= N, so A is never empty.
    try:
        A = list(map(int, sys.stdin.readline().split()))
    except (IOError, ValueError):
        return

    # 1. Create a sorted list of all 2N-K sock colors.
    # This can be built in O(N) time without a general sort.
    A_set = set(A)
    all_socks = []
    for i in range(1, N + 1):
        if i in A_set:
            all_socks.append(i)
        else:
            all_socks.append(i)
            all_socks.append(i)
    
    L = len(all_socks)

    # If there are fewer than 2 socks, no pairs can be formed.
    if L < 2:
        print(0)
        return

    # 2. Dynamic Programming approach
    # Let dp[i] be the minimum weirdness for the first i socks.
    dp = [0] * (L + 1)
    
    # Base cases: dp[0]=0, dp[1]=0 are handled by initialization.
    # The loop starts at i=2, correctly using dp[0].
    
    for i in range(2, L + 1):
        # Cost if we pair the last two socks (all_socks[i-2], all_socks[i-1])
        # and add to the optimal solution for the first i-2 socks.
        cost_if_pair_last_two = dp[i-2] + (all_socks[i-1] - all_socks[i-2])
        
        if i % 2 == 0:
            # For an even number of socks, all must be paired.
            # The i-th sock is optimally paired with the (i-1)-th.
            dp[i] = cost_if_pair_last_two
        else:
            # For an odd number, one is left over.
            # Option 1: Leave the i-th sock out. Cost is dp[i-1].
            # Option 2: Pair i-th and (i-1)-th. Cost is cost_if_pair_last_two.
            dp[i] = min(dp[i-1], cost_if_pair_last_two)
            
    # The final answer is the minimum weirdness for all L socks.
    print(dp[L])

solve()