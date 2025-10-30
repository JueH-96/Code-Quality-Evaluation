def main():
    import sys
    data = sys.stdin.read().strip().split()
    if not data:
        return
    it = iter(data)
    N = int(next(it))
    M = int(next(it))
    B = [int(next(it)) for _ in range(N)]
    mod = 998244353

    # Explanation:
    # We are given a sequence A and need to compute f(A), where f(A) is the number
    # of connected components in the graph built as follows:
    #   – vertices 1..N,
    #   – for every pair i<j, add an edge if A_i <= A_j.
    #
    # It turns out that one may prove that f(A) equals the number of "records" in
    # A when reading from right to left. In other words, if we define a right‐record
    # as an index i such that A_i > max(A_{i+1}, …, A_N) (with the last element always
    # being a record), then f(A) equals the number of right‐records. (See the examples below.)
    #
    # For example, for A = (2, 1, 1):
    #   – rightmost element (1) is always a record.
    #   – moving left, 1 is not greater than 1, but 2 > max(1,1).
    # Thus, f(A)=2.
    #
    # Our task is: given a partial sequence B (with some values fixed and -1 indicating a free spot,
    # where one can choose any value 1..M), sum f(B') over all completions B'.
    #
    # We now “transform” the problem to: For a complete sequence A, let
    #    f(A) = sum_{i=1}^N I( A_i is a right‐record ).
    # Then our answer is the sum of (record indicators over positions) on every complete assignment.
    #
    # We solve this with the common “digit DP” style going from right to left.
    # More precisely, let F(i, R) be a pair (S, C) defined as:
    #   – C = number of ways to assign positions i .. N (given the fixed entries),
    #   – S = total sum of record indicators from positions i .. N given that the maximum
    #         seen so far (i.e. among positions i+1..N) is R.
    #
    # In our recurrence, the decision at position i is: if B[i] is fixed then use that value,
    # else sum over v in 1..M.
    # For a chosen value v at position i, the record indicator is 1 if v > R, else 0.
    # The new "record" (i.e. current maximum for future positions) becomes max(R, v).
    #
    # Our final answer will be F(1, 0)_S (with initial R=0, less than any allowed value).
    #
    # Implementation details:
    # We use 0-indexing for positions in our code while our DP will run from i = N down to 1.
    # We use a 1D DP array for “state” parameter R (which can be 0...M; note R=0 means no value has been chosen yet).
    #
    # Let dp[r] = (S, C) be our DP for the suffix (for a fixed position i) if the maximum value from positions i+1..N is r.
    # Base case: for i = N+1 (i.e. when no positions remain),
    #   dp[r] = (0, 1) for every r (there is exactly one way to assign nothing and no record is contributed).
    #
    # Now process positions i from N down to 1.
    # If B[i] is fixed and equals v:
    #   new_max = max(r, v)
    #   record = 1 if v > r else 0
    #   dp[r] = ( record * (ways from dp[new_max]) + (sum from dp[new_max]),  dp[new_max].count )
    #
    # If B[i] is free (i.e. -1), we sum over v from 1 to M.
    # Notice that when v <= r, new_max remains r and record = 0.
    # When v > r, new_max becomes v and record = 1.
    #
    # Thus for free case with state r:
    #   dp[r].count = (number of choices where v<=r) + (sum_{v=r+1}^M (ways for new state v))
    #                 = (r * dp[r].count [for each v<=r, state remains r]) + (sum_{v=r+1}^M dp[v].count)
    #   dp[r].sum   = (for v<=r, no record, so 0 contribution) + sum_{v=r+1}^M (dp[v].sum + dp[v].count)
    #
    # To implement the free case efficiently we precompute suffix sums from the dp arrays.
    #
    # Finally, our answer is dp[0].sum (with initial maximum 0).
    #
    # Time complexity: There are O(N*M) states. A careful implementation with cumulative sums
    # in the free branch gives O(N*M). N,M <= 2000.
    
    # dp arrays: for each R in 0..M, we store two numbers: total record sum and total count.
    dp_sum = [0] * (M + 1)
    dp_cnt = [1] * (M + 1)  # Base: for i = N+1

    # Process positions from N down to 1 (here using 0-index: positions N-1 downto 0)
    for i in range(N - 1, -1, -1):
        new_dp_sum = [0] * (M + 1)
        new_dp_cnt = [0] * (M + 1)
        if B[i] != -1:
            v = B[i]
            # fixed case: for each possible current maximum r in 0..M.
            for r in range(M + 1):
                # new state will use w = max(r, v)
                w = r if r >= v else v
                # record contribution: 1 if v > r, else 0.
                rec = 1 if v > r else 0
                new_dp_sum[r] = (rec * dp_cnt[w] + dp_sum[w]) % mod
                new_dp_cnt[r] = dp_cnt[w] % mod
        else:
            # free case: v can be chosen from 1 to M.
            # For a given current maximum r:
            #   For v in 1..r: new state remains r, no record added.
            #   For v in r+1..M: new state becomes v, record added.
            # So,
            #   new_dp_cnt[r] = (r * dp_cnt[r]) + (sum_{v=r+1}^{M} dp_cnt[v])
            #   new_dp_sum[r] = (sum_{v=r+1}^{M} (dp_sum[v] + dp_cnt[v]))
            # To do these sums quickly for each r, we precompute suffix sums on dp_cnt and dp_sum+dp_cnt.
            suf_cnt = [0] * (M + 2)
            suf_sum = [0] * (M + 2)
            for r in range(M, -1, -1):
                suf_cnt[r] = (dp_cnt[r] + suf_cnt[r + 1]) % mod
                suf_sum[r] = (dp_sum[r] + dp_cnt[r] + suf_sum[r + 1]) % mod
            for r in range(M + 1):
                # For v in 1..r, there are exactly r choices and state remains r.
                part1 = (r * dp_cnt[r]) % mod
                # For v from r+1 to M: use suffix sum.
                part2 = suf_cnt[r + 1] % mod
                new_dp_cnt[r] = (part1 + part2) % mod
                new_dp_sum[r] = suf_sum[r + 1] % mod
        dp_sum, dp_cnt = new_dp_sum, new_dp_cnt
    # answer is dp[0].sum (with starting maximum 0)
    sys.stdout.write(str(dp_sum[0] % mod))
    
if __name__ == '__main__':
    main()