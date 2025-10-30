def main():
    import sys
    data = sys.stdin.read().split()
    if not data:
        return
    mod = 998244353
    n = int(data[0])
    k = int(data[1])
    arr = list(map(int, data[2:2+n]))
    
    # We use the following idea.
    # Let P[i] be the prefix sum for the first i elements with P[0] = 0.
    # A division of the array breaks A into segments. Each segment corresponds 
    # to a contiguous subarray and its sum equals P[j] - P[i] for some 0 <= i < j <= n.
    # We want to count the partitions such that for every segment (i+1..j), we have:
    #     P[j] - P[i] != k,  i.e.  P[i] != P[j] - k.
    #
    # Let dp[j] be the number of valid ways to partition the first j elements.
    # Then dp[0] = 1 (by convention) and for each j from 1 to n,
    # we can form a valid partition ending at j by choosing an earlier index i (0 <= i < j)
    # provided that the new segment sum P[j] - P[i] is not equal to k.
    # That is, dp[j] = sum_{i=0}^{j-1} dp[i], minus those dp[i]
    # for which P[i] equals (P[j] - k).
    #
    # To compute these sums efficiently we maintain:
    #   total = dp[0] + dp[1] + ... + dp[j-1]
    #   freq[s] = sum of dp[i] for i with P[i] = s.
    #
    # Then, when processing the j-th element (forming new prefix sum):
    #   dp[j] = total - freq.get(P[j] - k, 0)
    #
    # Finally, the answer will be dp[n] (which we build in one pass).
    
    total = 1  # This is sum(dp[0])
    dp_val = 0  # This will store dp for the current index.
    freq = {0: 1}  # Frequency map: for each prefix sum value, the cumulative dp sum.
    prefix = 0  # current prefix sum, starting with P[0] = 0.
    
    for x in arr:
        prefix += x
        # Exclude transitions i that would have produced a segment summing to k.
        # Such segments occur when P[i] == prefix - k.
        invalid = freq.get(prefix - k, 0)
        dp_val = (total - invalid) % mod
        # Now update total and freq dictionary.
        total = (total + dp_val) % mod
        freq[prefix] = (freq.get(prefix, 0) + dp_val) % mod

    sys.stdout.write(str(dp_val) + "
")

if __name__ == '__main__':
    main()