def main():
    import sys
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    it = iter(input_data)
    n = int(next(it))
    M = int(next(it))
    A = [int(next(it)) for _ in range(n)]

    # We will use two Fenwick trees (Binary Indexed Trees):
    #   one ("bit_f") to store frequency counts,
    #   and another ("bit_s") to store the sum of prefix mod M values.
    #
    # We define our prefix sum array P where:
    #   P[0] = 0,
    #   P[i] = (A[1] + A[2] + ... + A[i]) mod M  for 1 <= i <= n.
    #
    # Notice that for any contiguous subarray A[l...r] (with 1 <= l <= r),
    # we have:
    #   S = (P[r] - P[l-1]) mod M
    # which equals:
    #   if P[r] >= P[l-1]:  P[r] - P[l-1],
    #   else:              P[r] - P[l-1] + M.
    #
    # Thus, if we index our prefix mod M values as p0, p1, ..., p_n (with p0 = 0),
    # then the answer is:
    #     sum_{0 <= i < j <= n} f(p_j, p_i)
    # where:
    #     f(p_j, p_i) = { p_j - p_i            if p_i <= p_j,
    #                     p_j - p_i + M        if p_i > p_j }.
    #
    # To compute this fast, we process the prefix values in order, and for the current prefix value x = p_j,
    # we need to quickly calculate:
    #   - The sum over those previous prefix values p_i with p_i <= x:  count_less * x - sum_less.
    #   - And over those with p_i > x:  count_greater*(x + M) - sum_greater.
    # We then add both contributions.
    #
    # We will use BITs keyed by the value in the range 0 to M-1.
    
    size = M  # values range from 0 to M-1.
    bit_f = [0] * (size + 1)  # BIT for frequency counts.
    bit_s = [0] * (size + 1)  # BIT for sum of values.
    
    def bit_update(bit, i, delta):
        # BIT is 1-indexed; valid indices are 1 to size.
        while i <= size:
            bit[i] += delta
            i += i & -i

    def bit_query(bit, i):
        s = 0
        while i:
            s += bit[i]
            i -= i & -i
        return s
    
    # Initially, we have P[0] = 0.
    # We update the BITs at coordinate 0. (0 corresponds to BIT index 1)
    bit_update(bit_f, 1, 1)
    bit_update(bit_s, 1, 0)  # value is 0

    answer = 0
    prefix = 0
    # Process each A[i] to compute the new prefix and update our BITs.
    for i in range(n):
        prefix = (prefix + A[i]) % M
        x = prefix  # current prefix mod value
        
        # BIT index mapping: value x is stored at index x+1.
        # Query for values <= x.
        count_less = bit_query(bit_f, x + 1)
        sum_less = bit_query(bit_s, x + 1)
        # Total count and sum of already stored prefix values.
        total_count = bit_query(bit_f, size)
        total_sum = bit_query(bit_s, size)
        # The rest are prefix values > x.
        count_greater = total_count - count_less
        sum_greater = total_sum - sum_less
        
        # Contribution for the current prefix value:
        # For previous prefix values p:
        #   if p <= x, contribution is (x - p);
        #   if p > x, contribution is (x + M - p).
        contrib = count_less * x - sum_less + count_greater * (x + M) - sum_greater
        answer += contrib
        
        # Update BITs with current prefix value x.
        pos = x + 1
        bit_update(bit_f, pos, 1)
        bit_update(bit_s, pos, x)
    
    sys.stdout.write(str(answer))

if __name__ == '__main__':
    main()