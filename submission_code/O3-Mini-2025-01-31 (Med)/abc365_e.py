def main():
    import sys
    data = sys.stdin.buffer.read().split()
    if not data:
        return
    it = iter(data)
    n = int(next(it))
    A = [int(next(it)) for _ in range(n)]
    
    # We wish to compute the sum:
    #    S = sum_{i=1}^{N-1} sum_{j=i+1}^N (A_i XOR A_{i+1} XOR ... XOR A_j)
    #
    # A standard trick is to consider prefix XOR values.
    # Define B[0] = 0 and for 1 <= k <= N, let:
    #    B[k] = A_1 XOR A_2 XOR ... XOR A_k.
    # Then the XOR over a contiguous segment A_i ... A_j equals:
    #    B[j] XOR B[i-1].
    #
    # Hence the desired sum becomes:
    #    S = sum_{0 <= i < j <= N, and (j-i) >= 2} (B[i] XOR B[j])
    #
    # Notice that if j = i+1 then the segment has length 1.
    # The sum over all pairs 0 <= i < j <= N
    #    T = sum_{0 <= i < j <= N} (B[i] XOR B[j])
    # would include the segments of length 1.
    #
    # But note that for i = 0,1,...,N-1:
    #    B[i] XOR B[i+1] = A_{i+1}
    # So, the sum over segments of length 1 is simply sum_{i=1}^N A_i.
    #
    # Therefore, the answer is:
    #    Answer = T - sum_{i=1}^N A_i.
    #
    # We can efficiently compute T using a bitwise approach.
    #
    # For an array X of numbers, the sum over all pairs (i, j) with i < j:
    #    sum_{i < j} (X[i] XOR X[j])
    # can be computed for each binary bit independently.
    #
    # For a fixed bit position b, let:
    #    ones = number of X[i] (over all i) with the b-th bit set
    #    zeros = total count - ones
    # Then for every pair (i, j) with one value having bit b = 1 and the other 0,
    # the XOR contributes 2^b.
    # Since each pair (i,j) with i < j appears exactly once, the contribution for bit b is:
    #    contribution_b = (2^b) * (ones * zeros).
    #
    # Here our X is the prefix array B of length n+1.
    
    max_bit = 32  # Enough to cover A_i up to 10^8.
    total_prefixes = n + 1  # B has n+1 values.
    count_bit = [0] * max_bit  # count_bit[b] will count number of B's with bit b = 1.
    
    prefix = 0
    # B[0] = 0: update counts (0 has no bits set, so no update is required)
    # Now compute B[1] to B[n] and update counts.
    for x in A:
        prefix ^= x
        # Update bit counts for the current prefix value.
        for b in range(max_bit):
            if (prefix >> b) & 1:
                count_bit[b] += 1

    # We computed counts for B[1..n]. B[0] = 0 contributed zeros,
    # so total count for each bit remains unchanged (only B[1..n] matter).

    # Compute T = sum_{0 <= i < j <= n} (B[i] XOR B[j])
    # For B[0..n], since B[0] = 0, the count for each bit b is simply count_bit[b] (from indices 1..n)
    # and zeros = total_prefixes - count_bit[b].
    total_sum = 0
    # For each bit position, contribution is:
    #   2^b * (ones * zeros)
    for b in range(max_bit):
        ones = count_bit[b]
        zeros = total_prefixes - ones  # Since B[0] always has bit b = 0.
        total_sum += (1 << b) * ones * zeros

    # Subtract the contributions from segments of length 1.
    # For each i=1..n, B[i-1] XOR B[i] = A[i]. So subtract sum(A)
    sum_A = sum(A)
    ans = total_sum - sum_A
    sys.stdout.write(str(ans))
    
if __name__ == '__main__':
    main()