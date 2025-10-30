def solve():
    import sys
    import bisect
    
    input_data = sys.stdin.read().strip().split()
    N = int(input_data[0])
    T = int(input_data[1])
    S = input_data[2]
    X = list(map(int, input_data[3:]))

    # We want to count the number of passing pairs (i, j) with i<j in label.
    # A pair of ants (i, j) passes if and only if:
    #   - Their directions differ (one faces +1, the other faces -1).
    #   - They actually cross in time <= T+0.1.
    #
    # For crossing:
    #   if ant i is +1 and ant j is -1, then we need X_i < X_j (so i can catch j),
    #   and (X_j - X_i)/2 <= T+0.1  =>  X_j - X_i <= 2*(T+0.1).
    #
    #   if ant i is -1 and ant j is +1, then we need X_i > X_j (so j can catch i),
    #   and (X_i - X_j)/2 <= T+0.1  =>  X_i - X_j <= 2*(T+0.1).
    #
    # We will implement two passes in ascending/descending order of labels
    # to efficiently count these pairs using a Fenwick tree (Binary Indexed Tree).
    #
    # Let T' = 2*(T + 0.1) = 2T + 0.2.
    # We'll do coordinate-compression of all possible relevant X-coordinates.
    #
    # Pass A: Count pairs where i<j, S_i=1 (+), S_j=0 (-), X_i < X_j, and X_j - X_i <= T'.
    #         We will process the ants in ascending label order:
    #           - If we see an ant with direction=+1, we add its X to Fenwick tree.
    #           - If we see an ant with direction=-1, we query how many +1 ants
    #             have X in [X_j - T', X_j).
    #
    # Pass B: Count pairs where i<j, S_i=0 (-), S_j=1 (+), X_i > X_j, and X_i - X_j <= T'.
    #         This is equivalent to j>i, j facing +, i facing -, X_j < X_i <= X_j + T'.
    #         We can do a reverse-label pass:
    #           - If we see an ant with direction=-1, we add its X to Fenwick tree.
    #           - If we see an ant with direction=+1, we query how many -1 ants
    #             have X in [X_i - T', X_i).
    #
    # Each query is a range-count query in the compressed X domain.

    sys.setrecursionlimit(10**7)

    # Build an array of (label i, position X_i, direction s_i) to process in label order.
    ants = [(i+1, X[i], S[i]) for i in range(N)]
    # T' = 2T + 0.2
    Tprime = 2.0 * T + 0.2

    # We need coordinate compression over possible X-values that we might query:
    # For each X_i, we might query [X_i - Tprime, X_i) or similar for the other pass.
    # Collect all relevant coordinates into a list for compression.
    coords = []
    for i, pos, d in ants:
        coords.append(pos)
        # For queries, we need pos - Tprime and pos itself
        # (the upper bound in queries is pos, exclusive, but we still need it for indexing).
        coords.append(pos - Tprime)
        # We'll also store pos + Tprime sometimes, but strictly we only need
        # (pos - Tprime) and pos for the queries used in Fenwick range queries.
        # The upper bound "pos" is used with bisect_left(pos).
        # However, pos + Tprime might not be necessary in these exact queries,
        # so we won't add it to keep the coordinate set smaller.
        # If we needed it for some other approach we would add it.
    # Sort and unique
    coords = list(set(coords))
    coords.sort()

    # Fenwick (BIT) for counting how many ants lie at or below a certain compressed index
    class Fenwick:
        def __init__(self, n):
            self.n = n
            self.data = [0]*(n+1)
        
        def update(self, idx, val):
            # idx in [1..n] in Fenwick terms
            while idx <= self.n:
                self.data[idx] += val
                idx += idx & -idx
        
        def query(self, idx):
            # sum from 1..idx
            s = 0
            while idx > 0:
                s += self.data[idx]
                idx -= idx & -idx
            return s
        
        def range_query(self, l, r):
            if l>r:
                return 0
            return self.query(r) - self.query(l-1)

    def compress(x):
        # returns 1-based index in Fenwick sense
        # We'll use bisect_left on coords
        return bisect.bisect_left(coords, x) + 1

    fenw = Fenwick(len(coords))

    ans = 0

    # PASS A: (i < j) & (dir i= +, dir j= -), X_i < X_j <= X_i + T'
    # Sort ants by label ascending
    ants.sort(key=lambda v: v[0])  # v[0] is label i
    for i, pos, d in ants:
        if d == '1':
            # direction = + -> store it in Fenw.
            idx = compress(pos)
            fenw.update(idx, 1)
        else:
            # direction = - -> query how many plus-ants in [pos - Tprime, pos).
            left_val = pos - Tprime  # inclusive
            right_val = pos         # exclusive
            l_idx = bisect.bisect_left(coords, left_val) + 1
            r_idx = bisect.bisect_left(coords, right_val)  # -1 after that
            r_idx = r_idx if r_idx>0 else 0  # to stay non-negative
            r_idx = r_idx  # it's still 1-based after +1?
            # Careful: we used +1 in compress, so let's adapt carefully:
            # We want the largest index that is strictly less than right_val.
            # r_idx is the 1-based index for the left-bound of right_val.
            # So effectively, we'll do r_idx = compress(right_val) - 1
            # but let's replicate that logic:
            c_right = compress(right_val) - 1
            c_left = compress(left_val)
            if c_right >= c_left:
                ans += fenw.range_query(c_left, c_right)

    # PASS B: (i < j) & (dir i= -, dir j= +), X_j < X_i <= X_j + T'
    # Instead of "i<j in label order," we can process in descending label order
    # and interpret: if S_i='-', we store that position, if S_i='+',
    # then we query how many '-' are in [X_i - T', X_i).
    # Re-initialize Fenwicks for the second pass
    fenw = Fenwick(len(coords))

    # Sort ants by label descending
    ants.sort(key=lambda v: v[0], reverse=True) 
    for i, pos, d in ants:
        if d == '0':
            # direction = - -> store
            idx = compress(pos)
            fenw.update(idx, 1)
        else:
            # direction = + -> query how many minus-ants in [pos - Tprime, pos).
            left_val = pos - Tprime
            right_val = pos  # exclusive
            c_right = compress(right_val) - 1
            c_left = compress(left_val)
            if c_right >= c_left:
                ans += fenw.range_query(c_left, c_right)

    print(ans)