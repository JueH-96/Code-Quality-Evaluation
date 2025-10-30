def main():
    import sys
    data = sys.stdin.buffer.read().split()
    if not data:
        return
    it = iter(data)
    n = int(next(it))
    M = int(next(it))
    A = [int(next(it)) for _ in range(n)]
    
    # Compute prefix sums modulo M. Let B[0]=0, and for i>=1, 
    # B[i] = (A_1+...+A_i) mod M.
    B = [0]*(n+1)
    for i in range(1, n+1):
        B[i] = (B[i-1] + A[i-1]) % M

    # We need to sum over all subarrays l..r:
    #   ( (sum_{i=l}^{r} A[i]) mod M )
    #
    # Observing that the sum of the subarray is S = P[r]-P[l-1]
    # and (S mod M) = ( (P[r]-P[l-1]) mod M )
    # where P[i] is the normal prefix sum of A.
    # Equivalently, writing B[i] = (P[i] mod M), we have:
    #   (P[r]-P[l-1]) mod M = (B[r]-B[l-1] if B[r]>=B[l-1], else B[r]-B[l-1]+ M)
    #
    # We want to sum over all pairs (i, j) with 0 <= i < j <= n:
    #   f(B[j],B[i]) = { B[j]-B[i]   if B[i] <= B[j]
    #                   { B[j]-B[i]+M if B[i] > B[j]
    #
    # For fixed j (j from 1 to n), the contribution is:
    #   = sum_{i=0}^{j-1} (B[j]-B[i]) + M * (number of i with B[i]>B[j])
    #   = j*B[j] - (sum of B[i] for i=0..j-1) + M * (count_{i: B[i]>B[j]})
    #
    # We will compute these sums efficiently using Fenwick Trees (Binary Indexed Trees)
    # for counts and sums. The BIT indices will correspond to values 0..M-1.
    
    size = M  # Possible values: 0 to M-1.
    # BIT for counts (how many B[i] of a given value) and for sums.
    fenw_count = [0]*(size+1)
    fenw_sum = [0]*(size+1)
    
    # Standard update and query for BIT (Fenwick tree).
    def fenw_update(bit, i, delta):
        # i is 1-indexed.
        while i <= size:
            bit[i] += delta
            i += i & -i

    def fenw_query(bit, i):
        s = 0
        while i:
            s += bit[i]
            i -= i & -i
        return s
    
    # Helpers to update and query based on our B value.
    # We will store values at index (value+1) because BIT is 1-indexed.
    def update(v, count_delta, sum_delta):
        idx = v + 1  # convert v (in [0,M-1]) to BIT index.
        fenw_update(fenw_count, idx, count_delta)
        fenw_update(fenw_sum, idx, sum_delta)
    
    def query_count(v):
        # Returns number of indices with value <= v.
        return fenw_query(fenw_count, v+1)
    
    def query_sum(v):
        # Returns sum of values for indices with value <= v.
        return fenw_query(fenw_sum, v+1)
    
    total_answer = 0
    # Initialize BIT with the base prefix: B[0] = 0.
    update(0, 1, 0)
    
    # Process each prefix index j from 1 to n.
    for j in range(1, n+1):
        x = B[j]
        # Total number of previously seen prefix values (i from 0 to j-1).
        total_count = fenw_query(fenw_count, size)
        total_sum = fenw_query(fenw_sum, size)
        # Count of indices i with B[i] <= x.
        cnt_le = query_count(x)
        # For those with B[i] > x.
        cnt_greater = total_count - cnt_le

        # The contribution for fixed j is:
        #   j*x - total_sum + M * cnt_greater.
        contr = j * x + M * cnt_greater - total_sum
        total_answer += contr
        
        # Update BIT trees with the current value x.
        update(x, 1, x)
    
    sys.stdout.write(str(total_answer))
    
if __name__ == '__main__':
    main()