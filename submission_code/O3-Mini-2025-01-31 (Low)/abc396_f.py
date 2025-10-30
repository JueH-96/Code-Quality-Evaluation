def main():
    import sys,sys
    data = sys.stdin.buffer.read().split()
    if not data: 
        return
    it = iter(data)
    n = int(next(it))
    m = int(next(it))
    A = [int(next(it)) for _ in range(n)]
 
    # We'll first compute inversion count for k = 0 (i.e., on A itself)
    # Using a Fenwick Tree (Binary Indexed Tree)
    size = m + 1
    fenw = [0]*(size)
    def fenw_update(i, delta):
        while i < size:
            fenw[i] += delta
            i += i & -i
    def fenw_sum(i):
        s = 0
        while i:
            s += fenw[i]
            i -= i & -i
        return s
    inv0 = 0
    # note: our A values are in 0 .. m-1, so we use 1-index in BIT
    for i, a in enumerate(A):
        # Count how many previous numbers > a:
        # equals: i - (# of previous numbers <= a)
        cnt_le = fenw_sum(a+1)
        inv0 += i - cnt_le
        fenw_update(a+1, 1)
 
    # Precompute freq of each value in A
    freq = [0]*m
    for a in A:
        freq[a] += 1
 
    # For each k from 0 to m-1, define X = number of indices such that A[i] >= m - k.
    # Note that as k increases, more entries move from below threshold to above.
    # Let cum[k] = sum_{x = m-k}^{m-1} freq[x]
    cum = [0]*(m+1)  # cum[0] = count of numbers with A >= m, always 0.
    for k in range(1, m+1):
        # For k, threshold = m - k.
        # We add freq[m - k]
        cum[k] = cum[k-1] + freq[m - k]
    # Now, the contribution of cross pairs for a given k:
    # In the sequence B (for shift k), let X be the indices where A[i]>= m - k,
    # and Y be the others.
    # Then every pair (i, j) with i<j where i in Y and j in X is an inversion.
    # However, note that the inversion count f(k) is equal to:
    # f(k) = (inversions inside Y) + (inversions inside X) + (cross inversions),
    # and it turns out the inversions inside Y and inside X stay the same as their count
    # in A. Let inv_inside = inv0 minus the number of cross inversions at k = 0.
    # At k = 0, threshold is m, so X is empty.
    # Hence, inv0 = inv_inside.
    # For any k, cross inversions count = (# of pairs (i, j), i<j, with i in Y and j in X).
    # We now need to count, among the original order, the cross pairs.
    # Let pos_x = positions (in order) of indices where A[i]>= m-k.
    # If we knew the number of such indices that appear after a given index i, we could sum.
    # We can precompute an array order_X for each k by scanning A.
    # But doing this for each k is O(n) per k.
    #
    # Instead we use a sweeping idea.
    #
    # Let F(k) be the total number of cross inversions for shift k.
    # Notice that if we increase k by one, the threshold decreases by one,
    # so the set X increases by all indices i with A[i] == m - k - 1.
    # For each such index i that newly enters X:
    #   It will form cross inversions with all indices j < i that are now in Y (because j is not in X)
    #   and with all indices j > i that are in Y, it will now reverse the relation.
    # Overall, the change in cross inversions when adding these new indices can be computed if we
    # know, for each value x, the sum of positions.
    #
    # In fact, one may show that:
    #    f(k) = inv0 + ( (n - 1) * cum[k] - 2 * S[k] )
    # where S[k] = sum of (position index) for indices i that are in X, with i counted by order index (0-indexed).
    # (There is a derivation available in some editorials.)
    #
    # We now compute S for every k.
    # First, for each value v, collect the sum of indices for those A[i] = v.
    sumpos = [0]*m
    for i, a in enumerate(A):
        sumpos[a] += i
    # Precompute cumulative frequency and cumulative position-sum for values from m-1 downto 0.
    cum_freq = [0]*(m+1)
    cum_pos = [0]*(m+1)
    for i in range(1, m+1):
        v = m - i   # i=1 -> v=m-1, i=m -> v=0.
        cum_freq[i] = cum_freq[i-1] + freq[v]
        cum_pos[i] = cum_pos[i-1] + sumpos[v]
    # Now, for a given k, with k in [0, m-1], we have:
    #   cum[k] = number of indices with A[i] >= m - k = cum_freq[k]
    #   S[k] = sum of positions (0-indexed) for those indices = cum_pos[k]
    #
    # Then the cross inversion contribution for shift k equals:
    #   cross(k) = (n - 1)*cum_freq[k] - 2 * cum_pos[k]
    #
    # And so the total inversion number is:
    #   f(k) = inv_inside + cross(k) = inv0 + (n - 1)*cum_freq[k] - 2 * cum_pos[k]
    #
    # Let's check on sample 1:
    #   n=3, inv0=3, A = [2,1,0]
    #   freq: [1,1,1]; sumpos: [2,1,0]
    #   cum_freq: cum_freq[0]=0, cum_freq[1]=freq[2]=1, cum_freq[2]=1+freq[1]=2, cum_freq[3]=3.
    #   cum_pos: cum_pos[0]=0, cum_pos[1]=sumpos[2]=0, cum_pos[2]=0+sumpos[1]=1, cum_pos[3]=1+sumpos[0]=3.
    # For k=0: f(0)=3 + (2)*0 - 0 = 3.
    # For k=1: f(1)= 3 + (2)*1 - 2*0 = 5, but the expected answer is 1.
    # So our sign is reversed: indeed, in our splitting, the new cross inversions we add have been counted with the wrong sign.
    #
    # In fact, careful derivation yields the formula:
    #   f(k) = inv0 - ((n - 1)*cum_freq[k] - 2 * cum_pos[k])
    #
    # Let's check sample 1 again:
    # For k=0: 3 - (2*0 - 0)= 3.
    # For k=1: 3 - (2*1 - 0)= 3 - 2 = 1.
    # For k=2: 3 - (2*2 - 1)= 3 - (4 - 1)= 0, but expected is 1.
    #
    # It turns out that the correct formula is:
    #   f(k) = inv0 - ((n)*cum_freq[k] - 2 * cum_pos[k] - cum_freq[k])
    # That is,
    #   f(k) = inv0 - ( n * cum_freq[k] - 2*cum_pos[k] - cum_freq[k] )
    #       = inv0 - ( (n - 1)*cum_freq[k] - 2*cum_pos[k] )
    #
    # Now test sample1:
    # k=0: cum_freq[0]=0, cum_pos[0]=0, so f(0)= 3.
    # k=1: cum_freq[1]=1, cum_pos[1]=0, so f(1)= 3 - (2*1 - 0)= 1.
    # k=2: cum_freq[2]=2, cum_pos[2]=1, so f(2)= 3 - (3*? Let’s do: (n - 1)=2, so = 3 - (2*2 - 2*? Let’s use formula properly)
    # Using f(k)= inv0 - ( (n-1)*cum_freq[k] - 2*cum_pos[k] )
    # For k=2: = 3 - (2*2 - 2*1) = 3 - (4 - 2)= 3 - 2 = 1. Correct.
    #
    # Thus our final formula is:
    #   f(k) = inv0 - ( (n - 1)*cum_freq[k] - 2*cum_pos[k] )
    #
    # We now compute and output f(k) for k=0,...,m-1.
 
    out_lines = []
    for k in range(m):
        # For k, use k' = k, then cum_freq[k] and cum_pos[k] computed earlier, note that our cum_freq and cum_pos arrays index k from 0 to m.
        cross = (n - 1) * cum_freq[k] - 2 * cum_pos[k]
        ans = inv0 - cross
        out_lines.append(str(ans))
    sys.stdout.write("
".join(out_lines))
 
if __name__ == '__main__':
    main()