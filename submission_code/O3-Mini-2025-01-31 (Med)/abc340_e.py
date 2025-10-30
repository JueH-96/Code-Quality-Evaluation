def main():
    import sys,sys
    data = sys.stdin.read().strip().split()
    if not data:
        return
    it = iter(data)
    n = int(next(it))
    m = int(next(it))
    A = [int(next(it)) for _ in range(n)]
    # Operations: list of box indices from which to remove balls.
    B = [int(next(it)) for _ in range(m)]
    
    # We want to simulate M operations.
    # In each operation with box index b:
    #   Let Y be the current ball count in box b.
    #   Remove them from box b.
    #   Then distribute Y balls one-by-one to boxes:
    #       For k from 1 to Y, put a ball into box (b + k) mod n.
    # This process is equivalent to:
    #   1. Setting the ball count of box b to 0.
    #   2. Adding + floor(Y/n) to every box.
    #   3. And adding an extra +1 to all boxes in the contiguous cyclic block of length (Y mod n)
    #      starting from (b+1) mod n.
    #
    # However, since later operations can use the recently added balls,
    # we must simulate the operations sequentially.
    #
    # Note: An operation might add a full cycle (i.e. + (Y // n) to every box) and extra for a subinterval.
    #
    # We need to support:
    #   • Reading the current value in a given box b.
    #   • Updating each box’s value by a uniform addition (the full-cycle part)
    #     – We will maintain a variable "global_inc" for that.
    #   • Updating a cyclic subinterval (the extra +1’s) quickly.
    # We will use a Fenwick tree (Binary Indexed Tree) to support range updates (by adding a value on an interval)
    # and point queries (to know the extra addition for a box).
    #
    # Additionally, when we remove all balls from a box b, we want to “reset” that box.
    # We maintain an auxiliary array delta[] (for point corrections) so that the effective ball count at index i is:
    #     effective[i] = A[i] + global_inc + fenw_extra[i] + delta[i]
    # where A[i] is the initial (constant) contribution and fenw_extra[i] is the extra added via the Fenw tree.
    # When we remove balls from box b, letting Y be its current effective value,
    # we set its new delta[b] so that its effective value becomes 0.
    # That is, we set:
    #     A[b] + global_inc + fenw.query(b) + new_delta = 0   ==>   new_delta = -(A[b] + global_inc + fenw.query(b))
    # Since the effective ball count was Y = A[b] + global_inc + fenw.query(b) + (old delta[b]),
    # the removed balls amount is Y (regardless of the previous correction) and subsequent operations
    # will add new balls to box b via global and fenw updates.
    
    # Implement a Fenwick tree for point query / range update.
    class Fenw:
        __slots__ = ['n', 'tree']
        def __init__(self, n):
            self.n = n
            self.tree = [0] * (n + 1)
        def update(self, i, delta):
            # Standard BIT update; i is 0-indexed.
            i += 1
            while i <= self.n:
                self.tree[i] += delta
                i += i & -i
        def query(self, i):
            # Prefix sum from index 0 to i.
            s = 0
            i += 1
            while i:
                s += self.tree[i]
                i -= i & -i
            return s
        def range_update(self, l, r, delta):
            # Add delta to interval [l, r] (0-indexed).
            if l > r:
                return
            self.update(l, delta)
            if r + 1 < self.n:
                self.update(r + 1, -delta)
    
    fenw = Fenw(n)
    delta = [0] * n  # point adjustments so that effective[i] = A[i] + global_inc + fenw.query(i) + delta[i]
    global_inc = 0   # full-cycle additions applied uniformly
    
    # Process each operation sequentially.
    for b in B:
        # Compute the effective ball count in box b.
        cur = A[b] + global_inc + fenw.query(b) + delta[b]
        Y = cur
        # Remove all balls from box b by resetting its effective value to 0.
        # We set delta[b] so that A[b] + global_inc + fenw.query(b) + delta[b] becomes 0.
        delta[b] = - (A[b] + global_inc + fenw.query(b))
        
        if Y == 0:
            continue
        
        full = Y // n
        rem = Y % n
        global_inc += full  # add full-cycle to every box
        
        # Extra distribution: add +1 to the cyclic subinterval of length rem starting at (b+1) mod n.
        if rem:
            s = (b + 1) % n
            e = (b + rem) % n
            if s <= e:
                fenw.range_update(s, e, 1)
            else:
                # The interval wraps around: update from s to n-1 and 0 to e.
                fenw.range_update(s, n - 1, 1)
                fenw.range_update(0, e, 1)
    
    # Final effective ball count for each box.
    out = []
    for i in range(n):
        # effective value = A[i] + global_inc + fenw.query(i) + delta[i]
        val = A[i] + global_inc + fenw.query(i) + delta[i]
        out.append(str(val))
    sys.stdout.write(" ".join(out))
    
    
if __name__ == '__main__':
    main()