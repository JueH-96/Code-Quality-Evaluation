def main():
    import sys
    input_data = sys.stdin.read().strip().split()
    N, M = map(int, input_data[:2])
    P = list(map(int, input_data[2:2+N]))
    L = list(map(int, input_data[2+N:2+N+M]))
    D = list(map(int, input_data[2+N+M:]))

    # 1) Sum up all regular prices
    total_price = sum(P)

    # 2) Coordinate-compress all item prices (P) so we can efficiently "pick" items.
    #    Since P_i can be up to 1e9, we can't use a simple direct Fenwick tree of size 1e9.
    #    Instead we'll store unique prices in ascending order, and refer to them by rank.
    unique_prices = sorted(set(P))  # all distinct item prices in ascending order
    # Map from price to its compressed rank (1-based for Fenwick convenience)
    # rank_map[p] = index-in-unique_prices + 1
    rank_map = {}
    for i, val in enumerate(unique_prices):
        rank_map[val] = i + 1
    
    # 3) Count how many items of each rank we have
    max_rank = len(unique_prices)
    freq = [0]*(max_rank+1)
    for price in P:
        freq[rank_map[price]] += 1
    
    # 4) Fenwick Tree (Binary Indexed Tree) for frequencies
    #    We'll store freq[i] at Fenwicksum i; sum(i) gives how many items up to rank i remain unused.
    class Fenwick:
        def __init__(self, n):
            self.n = n
            self.data = [0]*(n+1)
        def update(self, i, v):
            while i <= self.n:
                self.data[i] += v
                i += i & -i
        def query(self, i):
            """Returns sum from 1..i."""
            s = 0
            while i > 0:
                s += self.data[i]
                i -= i & -i
            return s
        def build(self, arr):
            """Build Fenwick tree from freq array arr (1-based)."""
            for i in range(1, self.n+1):
                self.update(i, arr[i])
        def find_first_pos_with_prefix_over(self, base, start):
            """
            Find the smallest position p in [start..n] such that
            query(p) > base.  If none found, return -1.
            """
            # We're effectively searching for p where query(p) > base.
            # We also must have p >= start.
            # We'll do a standard Fenwick "binary search" for query(p).
            # But we must ensure p >= start. We'll do an offset approach:
            # base is the partial sum up to (start-1). We want the first p >= start with
            # query(p) > base.  We'll do query(start-1) to account, but simpler is:
            # let needed = base + 1. We want the smallest p with query(p) >= needed.
            needed = base + 1
            # first do a check if query(n) < needed => no such position
            if self.query(self.n) < needed:
                return -1
            
            # We'll do the usual Fenwick binary search for prefix >= needed.
            # Then we must check if p < start => we continue. So let's do
            # a search from 0..n. Then if we get p < start, we adjust upwards manually.
            pos = 0
            bit_mask = 1 << (self.n.bit_length())  # something >= n
            while bit_mask > 0:
                nxt = pos + bit_mask
                if nxt <= self.n and self.data[nxt] < needed:
                    needed -= self.data[nxt]
                    pos = nxt
                bit_mask >>= 1
            
            # pos will end up at the largest index where prefixSum <= base
            # so the answer is pos+1, but we also must ensure it's >= start
            candidate = pos + 1
            if candidate < start:
                # If candidate < start, we must re-search with a smaller range.
                # A simpler approach: we do base2 = self.query(start-1).
                # We want the first p >= start for which query(p) > base2.
                base2 = self.query(start-1)
                needed2 = base2 + 1
                # If query(n) < needed2 => no
                if self.query(self.n) < needed2:
                    return -1
                pos = 0
                bit_mask = 1 << (self.n.bit_length())
                while bit_mask > 0:
                    nxt = pos + bit_mask
                    if nxt <= self.n and self.data[nxt] < needed2:
                        needed2 -= self.data[nxt]
                        pos = nxt
                    bit_mask >>= 1
                candidate = pos + 1
            
            return candidate if candidate >= start else -1
    
    fenw = Fenwick(max_rank)
    fenw.build(freq)
    
    # 5) Sort coupons by discount descending
    coupons = sorted(zip(L, D), key=lambda x: x[1], reverse=True)
    
    total_discount = 0
    
    import bisect
    
    for l_i, d_i in coupons:
        # find the smallest rank whose price >= l_i
        # i.e. rank = bisect_left(unique_prices, l_i) + 1 (1-based)
        idx = bisect.bisect_left(unique_prices, l_i)
        if idx == len(unique_prices):
            # no item price >= l_i
            continue
        rank_needed = idx + 1  # 1-based rank
        # Now find the first actual rank >= rank_needed that is still available
        base = fenw.query(rank_needed - 1)  # sum of freq up to rank_needed-1
        pos = fenw.find_first_pos_with_prefix_over(base, rank_needed)
        if pos == -1:
            # no item found
            continue
        # Use that item
        fenw.update(pos, -1)  # remove one from freq for that rank
        total_discount += d_i
    
    # 6) Print the result
    print(total_price - total_discount)

# Don't forget to call main()
if __name__ == "__main__":
    main()