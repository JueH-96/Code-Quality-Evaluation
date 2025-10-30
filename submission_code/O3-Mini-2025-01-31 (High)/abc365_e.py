def main():
    import sys
    data = sys.stdin.buffer.read().split()
    if not data:
        return
    # Read N and the sequence A.
    N = int(data[0])
    A = [int(x) for x in data[1:1+N]]
    
    # Compute the prefix XOR array Q of length N+1.
    # We define Q[0] = 0 and for 1 <= i <= N, Q[i] = A[0] xor A[1] xor ... xor A[i-1]
    Q = [0] * (N + 1)
    for i in range(N):
        Q[i+1] = Q[i] ^ A[i]
    
    # Our goal is to compute:
    #    S = sum_{1 <= i < j <= N} (A_i xor A_{i+1} xor ... xor A_j)
    # Rewriting each subarray XOR in terms of Q, note that:
    #    A_i xor ... xor A_j = Q[i-1] xor Q[j]
    # and valid subarrays correspond to index pairs (x, y) = (i-1, j) with 0 <= x < y <= N
    # and with the extra condition that j - x >= 2 (since i-1 and j must be at least 2 apart).
    # Observe that for any consecutive indices x and x+1 we get subarrays of length 1,
    # so the valid pairs (x,y) we sum over are exactly all pairs with x < y
    # excluding the consecutive ones.
    #
    # To sum these XOR values bit‐by–bit, we note that the b–th bit of the XOR of Q[x] and Q[y]
    # is 1 exactly when the b–th bits of Q[x] and Q[y] differ.
    # Thus, if we could count the number of pairs (x,y) with 0 <= x < y <= N and y - x >= 2
    # such that the b–th bits differ, the b–th bit contributes (2^b) times that count.
    #
    # For an array of bits, a standard trick is:
    #   Number of unordered pairs with one 0 and one 1 = (# zeros) * (# ones)
    # And notice that each unordered pair gives exactly 1 valid ordering (i.e. with x < y).
    #
    # But here the summation is over all pairs (x,y) with x < y in Q,
    # from which we must subtract the ones that come from consecutive indices.
    # 
    # IMPORTANT: For consecutive indices, we have:
    #    Q[x+1] = Q[x] xor A[x]; hence the b–th bit changes (i.e. Q[x] and Q[x+1] differ)
    # exactly when the b–th bit of A[x] is 1.
    #
    # So for each bit b, let:
    #   total_diff = (# indices with Q[...] having 0 in bit b) * (# with 1 in bit b)
    # and
    #   invalid = (number of indices i from 0 to N-1 with A[i]'s b–th bit = 1)
    # Then the count of valid (non‐consecutive) pairs in Q with a difference at bit b is:
    #   valid_pairs = total_diff - invalid
    #
    # Finally, the answer equals:
    #    sum_{b=0}^{31} (valid_pairs * 2^b)
    
    L = N + 1  # length of Q
    ans = 0
    for b in range(32):
        mask = 1 << b
        # Count how many Q's have the b-th bit equal to 1.
        cnt1 = sum(1 for q in Q if q & mask)
        cnt0 = L - cnt1  # those with the b-th bit zero
        # Total (unordered) pairs with different bits (each unordered pair corresponds to one valid i<j pair).
        diff_pairs = cnt0 * cnt1
        # Count invalid consecutive pairs. For consecutive Q, Q[i+1] = Q[i] ^ A[i],
        # so the b–th bit changes if and only if A[i] has the b–th bit set.
        invalid = sum(1 for a in A if a & mask)
        valid_pairs = diff_pairs - invalid
        ans += valid_pairs * mask

    sys.stdout.write(str(ans))

if __name__ == '__main__':
    main()