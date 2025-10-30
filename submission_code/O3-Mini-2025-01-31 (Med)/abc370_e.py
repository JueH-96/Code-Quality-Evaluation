def main():
    import sys
    from collections import defaultdict
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    mod = 998244353  # Modulus defined by problem
    it = iter(input_data)
    N = int(next(it))
    K = int(next(it))
    A = [int(next(it)) for _ in range(N)]
    
    # The key idea:
    # We view a valid division as choosing cut positions between consecutive elements.
    # Let dp[j] be the number of ways to partition the first j elements (with indices 1..j) 
    # such that none of the segments has a sum equal to K. We also define an extra base state dp[0] = 1.
    #
    # Let P[i] be the prefix sum P[i] = A_1 + A_2 + ... + A_i, with P[0] = 0.
    # Then the sum of a segment that starts immediately after a previous cut at index i (i.e. state i)
    # and ends at j is: P[j] - P[i].
    # We want that for any segment we pick, it holds that P[j] - P[i] != K.
    # Hence, when forming a new segment from a state with prefix sum P[i], we have to avoid that
    # P[j] = P[i] + K.
    #
    # If we are going to end a segment at index j, then the number of ways dp[j] is essentially 
    # the sum of dp[i] over all previous positions i (i from 0 to j-1) such that the segment
    # from i+1 to j is not "bad", i.e., such that P[j] - P[i] != K.
    #
    # This recurrence can be written as:
    #   dp[j] = (sum_{0 <= i < j} dp[i]) - (sum_{i with P[i] == P[j] - K} dp[i])
    # Let prefixDP[j-1] = dp[0] + dp[1] + ... + dp[j-1]. Then:
    #   dp[j] = prefixDP[j-1] - (sum_{i with P[i] == P[j] - K} dp[i]).
    #
    # To compute this quickly we will maintain:
    #   - a running total prefixDP = dp[0] + ... + dp[j-1]
    #   - a dictionary (sumForVal) mapping a prefix sum value to the sum of dp[i] for indices i with that prefix sum.
    
    dp = [0] * (N + 1)
    dp[0] = 1
    prefixDP = 1  # Initially dp[0]
    
    sumForVal = defaultdict(int)
    curr_prefix = 0
    sumForVal[curr_prefix] = 1  # P[0] = 0 contributes dp[0]
    
    # Process positions 1 through N; A[j-1] is the j-th element.
    for j in range(1, N + 1):
        curr_prefix += A[j - 1]  # This is P[j]
        # dp[j] is computed from all ways to end the previous segment before j:
        #   dp[j] = prefixDP (over i from 0 to j-1) minus the ways that would lead to a segment summing exactly to K.
        subtract_val = sumForVal.get(curr_prefix - K, 0)
        dp_val = (prefixDP - subtract_val) % mod
        dp[j] = dp_val
        prefixDP = (prefixDP + dp_val) % mod
        sumForVal[curr_prefix] = (sumForVal[curr_prefix] + dp_val) % mod
        
    sys.stdout.write(str(dp[N] % mod))
    
if __name__ == '__main__':
    main()