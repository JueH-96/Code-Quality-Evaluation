def main():
    import sys
    data = sys.stdin.read().split()
    if not data:
        return
    it = iter(data)
    n = int(next(it))
    m = int(next(it))
    # initial balls in boxes numbered 0 to n-1.
    init = [int(next(it)) for _ in range(n)]
    # each operation: choose box b from which we remove all balls
    B = [int(next(it)) for _ in range(m)]
    
    # We use a Fenwick tree (Binary Indexed Tree) that supports range updates and point queries.
    # We work with 0-indexed positions outside the BIT but the BIT itself is built with 1-indexing.
    class Fenw:
        def __init__(self, n):
            self.n = n
            self.bit = [0] * (n + 1)
        
        def update(self, i, delta):
            # i is 1-indexed.
            while i <= self.n:
                self.bit[i] += delta
                i += i & -i
        
        def query(self, i):
            # query on 1-indexed position i
            s = 0
            while i:
                s += self.bit[i]
                i -= i & -i
            return s
        
        def range_update(self, l, r, delta):
            # l and r are 0-indexed; add delta to each position in [l, r]
            self.update(l + 1, delta)
            if r + 1 < self.n:
                self.update(r + 2, -delta)
    
    fenw = Fenw(n)
    # Initialize the BIT with the initial values.
    for i, val in enumerate(init):
        fenw.range_update(i, i, val)
    
    # Process each operation in order.
    # Operation: for chosen box b,
    #  1. let x = current number of balls in box b.
    #  2. Remove all x balls from box b.
    #  3. Distribute these x balls one-at-a-time: that means each ball goes to box ((b + step) mod n)
    #     where step goes from 1 to x.
    # Instead of simulating ball-by-ball (which is too slow), we note:
    #   Every box gets floor(x/n) balls (because we complete x//n full cycles)
    #   Then the first (x mod n) boxes in the sequence (starting at b+1 with wrap-around) get 1 extra ball.
    for b in B:
        # get the current number of balls in box b.
        current = fenw.query(b + 1)
        if current == 0:
            # nothing to do if the box is empty
            continue
        # Remove all balls from box b.
        fenw.range_update(b, b, -current)
        # Calculate how many full cycles and remainder steps there are.
        full = current // n
        rem = current % n
        if full:
            # uniform addition: every box gets full extra balls.
            fenw.range_update(0, n - 1, full)
        if rem:
            # for the remainder, add 1 ball to boxes starting from (b+1) to (b+rem) modulo n.
            start = (b + 1) % n
            end = (b + rem) % n
            if start <= end:
                fenw.range_update(start, end, 1)
            else:
                # when the range wraps around the end of the list.
                fenw.range_update(start, n - 1, 1)
                fenw.range_update(0, end, 1)
    
    # After all operations, output the final number of balls in each box.
    result = []
    for i in range(n):
        result.append(str(fenw.query(i + 1)))
    sys.stdout.write(" ".join(result))

if __name__ == '__main__':
    main()