def main():
    import sys
    input_data = sys.stdin.read().strip().split()
    # Faster pointer-based parsing
    idx = 0
    N = int(input_data[idx]); idx+=1
    M = int(input_data[idx]); idx+=1
    
    A = list(map(int, input_data[idx:idx+N]))
    idx += N
    
    B = list(map(int, input_data[idx:idx+M]))
    idx += M
    
    # ----------------------------------------------------------------------
    # Because X (the number of balls taken from a box) can be very large (up to 1e9),
    # and M can be up to 2e5, we cannot simulate each ball individually.
    #
    # Key observation:
    #   If we remove X balls from box b, then place them one by one into consecutive
    #   boxes (b+1, b+2, ..., b+X) mod N, this is the same as:
    #     1) Removing X from box b (new box b = old box b - X).
    #     2) Distributing X among the boxes in a circular fashion. Each of the X boxes
    #        in that sequence gets +1.
    #
    #     But if X >= N, we actually wrap around multiple times. Specifically:
    #        - Each of the N boxes gets floor(X / N) balls.
    #        - Then the first (X mod N) boxes after b (clockwise) get 1 more ball each.
    #
    # We must process M operations in order, and after each operation, the distribution
    # changes, affecting subsequent operations. We need an efficient data structure
    # that supports:
    #   - Querying the current value of a single box b (to find X).
    #   - Setting that box to 0 (removing X).
    #   - Adding a constant to all boxes.
    #   - Adding +1 to a contiguous segment in the circular array.
    #
    # One way:
    #   - Keep a global offset "globalAdd" that tracks how much has been added to *every* box.
    #   - Maintain a Fenwick (Binary Indexed) Tree (or Segment Tree) that stores the "extra"
    #     above (or below) that global offset for each box.
    #   - Then the actual number of balls in box i is: globalAdd + fenwicksum(i).
    #
    # Operations:
    #   - To find the current number in box b:  X = globalAdd + fenwicksum(b).
    #   - To set box b to 0: we want newExtra[b] = -globalAdd, so that
    #       globalAdd + newExtra[b] = 0.
    #     Hence the point update is:  delta = (-globalAdd) - oldExtra[b].
    #   - To add "val" to all boxes:  globalAdd += val.
    #   - To add +1 to [p..q] in circular manner (if p<=q), do a Fenwick range update of +1.
    #
    # Finally, after all M operations, box i has  globalAdd + fenwicksum(i).
    #
    # We'll implement a Fenwick tree that supports:
    #   - range_update(l, r, val): add "val" to each index in [l,r] (0-based).
    #   - point_query(i): return the value at index i.
    #
    # Standard trick for Fenwicks with range updates + point queries uses
    # two Fenwicks (bit1, bit2) and the identity:
    #   value(i) = i * sum(bit1, i) - sum(bit2, i)   (for 1-based i).
    #
    # We'll wrap these in 0-based calls for convenience.
    #
    # Complexity:
    #   - We do N initial updates to set the starting array (in Fenwicks).
    #   - Then M operations. Each operation involves:
    #       1 point query (O(log N))
    #       up to 1 point update (O(log N)) to remove X,
    #       1 globalAdd update (O(1)),
    #       up to 2 range updates (O(log N) each) to place the leftover (X mod N).
    #     So ~4 Fenwicks calls per operation => 4*M*logN, plus N logN for initialization.
    #   - This should be acceptable for N,M up to 2e5 if implemented efficiently in Python.
    # ----------------------------------------------------------------------
    
    # Fenwicks with range-update, point-query (using 1-based internally).
    class Fenwicks:
        __slots__ = ['n','bit1','bit2']
        def __init__(self, n):
            self.n = n
            # 1-based indexing, so allocate n+1
            self.bit1 = [0]*(n+1)
            self.bit2 = [0]*(n+1)
        
        def _update(self, bit, i, val):
            while i <= self.n:
                bit[i] += val
                i += (i & -i)
        
        def _sum(self, bit, i):
            s = 0
            while i > 0:
                s += bit[i]
                i -= (i & -i)
            return s
        
        # range update [l, r] with value val, 0-based
        def range_update(self, l, r, val):
            # internally shift +1 for 1-based
            l1 = l+1
            r1 = r+1
            # update bit1
            self._update(self.bit1, l1, val)
            self._update(self.bit1, r1+1, -val)
            # update bit2
            self._update(self.bit2, l1, val*(l))
            self._update(self.bit2, r1+1, -val*(r))
        
        # get the Fenwicks-based "prefix sum" up to index i (1-based)
        # then transform to get the actual value at index i
        def point_query(self, i):
            # i is 0-based
            i1 = i+1
            # formula: val(i) = i * sum(bit1, i) - sum(bit2, i)
            return i * self._sum(self.bit1, i1) - self._sum(self.bit2, i1)
    
    # Initialize Fenwicks structure
    fenw = Fenwicks(N)
    
    # We'll store "extra[i]" in Fenwicks, and maintain globalAdd separately.
    globalAdd = 0
    
    # Build initial distribution into fenw by doing point updates: range_update(i,i,A[i]).
    # This is O(N log N).  (We could do a more advanced build in O(N) but we'll keep it simpler.)
    range_update = fenw.range_update
    point_query   = fenw.point_query
    
    # Populate Fenwicks with the initial A.
    # point_update(i, val) = range_update(i, i, val).
    for i in range(N):
        if A[i] != 0:
            range_update(i, i, A[i])
    
    # Now process each operation
    # For each B_i:
    #   X = globalAdd + point_query(B_i)
    #   set box B_i to 0 => new extra[B_i] = -globalAdd
    #   so we do delta = -globalAdd - oldExtra.  oldExtra = point_query(B_i)
    #   full = X // N
    #   r    = X % N
    #   globalAdd += full
    #   then add +1 to the first r boxes after B_i: [B_i+1 .. B_i+r] mod N
    #
    # Watch for the case r=0 => no partial segment.
    # Also if r>0 and p>q => we do two range updates.
    
    for b_i in B:
        # how many are currently in box b_i?
        oldExtra = point_query(b_i)     # extra part
        X = globalAdd + oldExtra        # total in box b_i
        
        # remove X from box b_i => new value for box b_i is 0
        # we want fenwicks' extra for b_i to become -globalAdd
        delta = (-globalAdd) - oldExtra
        if delta != 0:
            range_update(b_i, b_i, delta)
        
        # now distribute X
        if X > 0:
            full = X // N
            r = X % N
            globalAdd += full
            if r > 0:
                start = (b_i + 1) % N
                end = (b_i + r) % N   # inclusive
                if start <= end:
                    range_update(start, end, 1)
                else:
                    # wrap around
                    range_update(start, N-1, 1)
                    range_update(0, end, 1)
    
    # After all M operations, compute final distribution:
    # box i = globalAdd + fenw.point_query(i).
    ans = []
    for i in range(N):
        val = globalAdd + point_query(i)
        ans.append(str(val))
    
    print(" ".join(ans))