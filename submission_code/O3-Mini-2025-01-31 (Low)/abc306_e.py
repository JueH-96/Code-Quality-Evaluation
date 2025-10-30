def main():
    import sys, bisect
    data = sys.stdin.read().split()
    if not data:
        return
    it = iter(data)
    N = int(next(it))
    K = int(next(it))
    Q = int(next(it))
    updates = []
    for _ in range(Q):
        x = int(next(it))
        y = int(next(it))
        updates.append((x, y))
        
    # We will use the idea that f(A) is the sum of the top K largest values.
    # Since all A_i sum to total_sum, notice that:
    #   f(A) = total_sum - (sum of the smallest N-K values)
    # So, if we can quickly compute the sum of the smallest N-K values after each update,
    # then our answer is readily available.
    #
    # Since Q and N are large, we cannot sort after each update.
    # We will use two Fenwick trees (Binary Indexed Trees) over the coordinate-compressed value space.
    # One Fenwick (fenw_count) to track the frequency of each value,
    # and another (fenw_sum) to track the total sum contributed by each value.
    #
    # Then, the total sum is maintained as updates occur.
    # To compute the sum of the smallest m = N-K values, we need to find (by binary search on fenw_count)
    # the last bucket that contributes partially and then compute the prefix sum.
    #
    # Note that we need to know all possible values that A[i] might take (initially 0, and then Y_i),
    # so that we can coordinate-compress them.
    
    # Collect all possible values – initial 0 and every update's y.
    vals = set()
    vals.add(0)
    for (_, y) in updates:
        vals.add(y)
    comp = sorted(vals)
    m = len(comp)
    # Map each value to its 1-indexed coordinate (for Fenwick tree purposes)
    comp_idx = {val: i+1 for i, val in enumerate(comp)}
    
    # Fenwick tree implementation (1-indexed)
    class Fenw:
        __slots__ = ['n', 'tree']
        def __init__(self, n):
            self.n = n
            self.tree = [0] * (n + 1)
        def update(self, i, delta):
            while i <= self.n:
                self.tree[i] += delta
                i += i & -i
        def query(self, i):
            s = 0
            while i:
                s += self.tree[i]
                i -= i & -i
            return s
        # find the smallest index such that the prefix sum is >= target.
        def lower_bound(self, target):
            idx = 0
            bit = 1 << (self.n.bit_length() - 1)
            while bit:
                nxt = idx + bit
                if nxt <= self.n and self.tree[nxt] < target:
                    target -= self.tree[nxt]
                    idx = nxt
                bit //= 2
            return idx + 1

    # Build two Fenwicks: one for counts and one for sums.
    fenw_count = Fenw(m)
    fenw_sum = Fenw(m)
    
    # Initially, all N elements are 0.
    zero_idx = comp_idx[0]
    fenw_count.update(zero_idx, N)
    # Since value is 0, sum update does nothing.
    
    # Maintain current array A (1-indexed for convenience).
    A = [0] * (N + 1)
    total_sum = 0  # total sum over all elements.
    
    # m_small is the number of smallest values to exclude (i.e. remove) from the total sum to
    # get the sum of the top K largest.
    m_small = N - K
    
    # Process each update.
    output_lines = []
    for (x, y) in updates:
        # Remove the old value and update the new value.
        old_val = A[x]
        A[x] = y
        old_idx = comp_idx[old_val]
        fenw_count.update(old_idx, -1)
        fenw_sum.update(old_idx, -old_val)
        new_idx = comp_idx[y]
        fenw_count.update(new_idx, 1)
        fenw_sum.update(new_idx, y)
        total_sum += (y - old_val)
        
        # Compute f(A) = total_sum - (sum of smallest m_small values)
        if m_small <= 0:
            # If no elements need to be removed, answer is total sum.
            ans = total_sum
        else:
            # Find the cumulative sum of the smallest m_small numbers using the fenwicks.
            # Identify the smallest value v (by its bucket t) such that the prefix of counts reaches m_small.
            t = fenw_count.lower_bound(m_small)
            # Query the prefix sums (for counts and for sum) up to index t-1.
            count_before = fenw_count.query(t - 1)
            sum_before = fenw_sum.query(t - 1)
            # The remaining count that falls in bucket t:
            rem = m_small - count_before
            # Actual value corresponding to bucket t:
            val_t = comp[t - 1]
            removal = sum_before + rem * val_t
            ans = total_sum - removal
        output_lines.append(str(ans))
    sys.stdout.write("
".join(output_lines))

if __name__ == '__main__':
    main()