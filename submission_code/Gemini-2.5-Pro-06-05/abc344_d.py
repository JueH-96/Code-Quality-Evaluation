# YOUR CODE HERE
import sys

def solve():
    """
    Solves the problem by reading from stdin and writing to stdout.
    """
    try:
        T = sys.stdin.readline().strip()
        if not T:
            return
        
        N_str = sys.stdin.readline().strip()
        if not N_str:
            # Handle case where N is missing but T is not.
            # If T is not empty, it's impossible to form without bags.
            print(-1)
            return
        N = int(N_str)
        
        bags = [sys.stdin.readline().split()[1:] for _ in range(N)]
    except (IOError, ValueError, IndexError):
        # Graceful exit on malformed input.
        # For an empty target string, the cost is 0, but constraints say len(T) >= 1.
        # If input is truly empty, this will prevent errors.
        return

    L = len(T)

    # dp[j] will store the minimum cost to form the prefix T[:j].
    # The maximum possible cost is N, so N+1 serves as a representation for infinity.
    inf = N + 1
    dp = [inf] * (L + 1)

    # Base case: It costs 0 to form an empty string (prefix of length 0).
    dp[0] = 0

    # Iterate through each bag and update the DP table.
    for bag in bags:
        # We iterate j (the length of the target prefix) downwards.
        # This is a space-optimized DP approach. When we calculate the new value for dp[j],
        # we use values like dp[j - len(s)], which, due to the downward iteration,
        # still hold the values from the previous step (before processing this bag).
        # The "do nothing" option for the current bag is implicitly handled, as dp[j]
        # retains its value from the previous step unless a cheaper path is found.
        for j in range(L, -1, -1):
            for s in bag:
                len_s = len(s)
                # Check if we can form T[:j] by appending string s.
                # This requires that the prefix T[:j-len_s] was reachable
                # and that s matches the suffix of T[:j].
                if j >= len_s and T[j - len_s:j] == s:
                    if dp[j - len_s] != inf:
                        dp[j] = min(dp[j], dp[j - len_s] + 1)

    # The final answer is the minimum cost to form the entire string T.
    ans = dp[L]

    # If the cost is still infinity, it means T cannot be formed.
    if ans == inf:
        print(-1)
    else:
        print(ans)

solve()