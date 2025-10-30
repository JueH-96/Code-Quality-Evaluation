# YOUR CODE HERE
def main():
    import sys,sys
    data = sys.stdin.read().split()
    if not data:
        return
    it = iter(data)
    try:
        n = int(next(it))
        m = int(next(it))
    except StopIteration:
        return
    A = [int(next(it)) for _ in range(n)]
    
    # We'll implement a Fenwick tree (binary indexed tree)
    class Fenw:
        __slots__ = "n", "data"
        def __init__(self, n):
            self.n = n
            self.data = [0]*(n+1)
        def update(self, i, delta):
            # i is 0-indexed.
            i += 1
            while i <= self.n:
                self.data[i] += delta
                i += i & -i
        def query(self, i):
            s = 0
            i += 1
            while i:
                s += self.data[i]
                i -= i & -i
            return s

    fenw = Fenw(m)
    # freq array for counts (for “equal” queries)
    freq = [0]*m

    # D (difference array) for offsets 0..m.
    D = [0]*(m+1)
    # left-to–right pass:
    # For each index j (0-indexed) let y = A[j]. Among the j values already seen:
    #   Let L = count of values < y,
    #   Let E = count of values equal to y.
    # Then total_previous = j, and G = (j - L - E) is count of values > y.
    # For each pair (i,j) then:
    #   - if A[i] < y, we “add” +1 at D[m-y]
    #   - if A[i] > y, we “add” +1 at D[0] and +1 at D[m-y]
    # So the combined update for index j is:
    #   D[m - y] += (j - E)   and   D[0] += G.
    for j in range(n):
        y = A[j]
        total_prev = j
        if y > 0:
            L = fenw.query(y-1)
        else:
            L = 0
        E = freq[y]
        G = total_prev - L - E
        D[m - y] += (total_prev - E)
        D[0] += G
        fenw.update(y, 1)
        freq[y] += 1

    # Now subtract the “first–element” contributions.
    # For each index i (as first element of a pair) with A[i] = v,
    # the number of pairs in which this index appears (with a later element)
    # and where the later element has a different value is:
    #      R_i = ( (n-1 - i) - ( (# of j>i with A[j]==v) ) ).
    # We subtract R_i from D[m-v].
    total_occ = [0]*m
    for v in A:
        total_occ[v] += 1
    count_seen = [0]*m
    for i in range(n):
        v = A[i]
        count_seen[v] += 1  # count (including current)
        # Number of later indices j > i with A[j] equal to v:
        same_after = total_occ[v] - count_seen[v]
        R = (n - 1 - i) - same_after
        D[m - v] -= R

    # Now F(k) = prefix sum of D[0]..D[k] for 0 <= k < m.
    out_lines = []
    curr = 0
    for k in range(m):
        curr += D[k]
        out_lines.append(str(curr))
    sys.stdout.write("
".join(out_lines))

if __name__ == '__main__':
    main()