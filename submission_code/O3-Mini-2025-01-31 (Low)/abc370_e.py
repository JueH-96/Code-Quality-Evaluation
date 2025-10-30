def main():
    import sys
    from collections import defaultdict
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    mod = 998244353
    it = iter(input_data)
    n = int(next(it))
    K = int(next(it))
    A = [int(next(it)) for _ in range(n)]
    
    # We have an array A of length n.
    # Let P[i] be the prefix sum up to index i: P[0]=0; and for 1<=i<=n, P[i]=A1+A2+...+Ai.
    #
    # Any division is encoded by indices 0 = j0 < j1 < ... < jk = n.
    # The segment (j_{r}+1 ... j_{r+1}) has sum P[j_{r+1}] - P[j_r].
    # We do not want any segment with sum exactly K.
    # That is, for every chosen boundary pair (j, i) with j < i, we must have:
    #   P[i] - P[j] != K
    # which is equivalent to 
    #   P[j] != P[i] - K.
    #
    # Define dp[i] as the number of ways to partition A[1..i] (i from 0...n)
    # such that no segment's sum equals K and boundaries always occur at the prefix
    # positions (with dp[0]=1).
    #
    # To transition to dp[i] we consider ending the last segment at i.
    # For every j with 0<= j < i, if the segment A[j+1...i] is valid then we add dp[j].
    # The condition is that P[i] - P[j] != K, i.e. P[j] != P[i]-K.
    #
    # Hence, we would have:
    #   dp[i] = total(dp[0] + ... + dp[i-1]) - (sum of dp[j] for which P[j] == P[i]-K)
    #
    # We'll compute the prefix sums on the fly with dp[0] = 1.
    #
    # We also maintain:
    #  total = dp[0] + dp[1] + ... + dp[i-1]
    #  dictionary mapping each prefix sum value P[j] to the sum of dp[j]
    
    dp = [0] * (n + 1)
    dp[0] = 1
    total = 1  # total sum of dp[0..i-1]
    
    # dictionary: key = prefix sum value, value = cumulative dp-sum for indices j with that prefix sum.
    prefix_dict = defaultdict(int)
    prefix_dict[0] = 1  # P[0] is 0 with dp[0]=1
    
    current_prefix = 0
    for i in range(1, n + 1):
        current_prefix += A[i - 1]  # P[i] = current_prefix
        # For a valid segmentation ending at i, last chosen boundary j must have
        # P[j] != current_prefix - K. The sum contributed from bad j is prefix_dict[current_prefix - K].
        bad = prefix_dict.get(current_prefix - K, 0)
        ways = (total - bad) % mod
        dp[i] = ways
        total = (total + ways) % mod
        prefix_dict[current_prefix] = (prefix_dict[current_prefix] + ways) % mod

    sys.stdout.write(str(dp[n] % mod))

if __name__ == "__main__":
    main()