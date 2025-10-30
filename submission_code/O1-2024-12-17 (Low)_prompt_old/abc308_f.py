def solve():
    import sys
    input_data = sys.stdin.read().strip().split()
    N, M = map(int, input_data[:2])
    P = list(map(int, input_data[2:2+N]))
    L = list(map(int, input_data[2+N:2+N+M]))
    D = list(map(int, input_data[2+N+M:]))

    # ---------------------------------------------------------
    # 1) Sum all item prices; we'll subtract discounts as we use coupons
    # ---------------------------------------------------------
    total_cost = sum(P)

    # ---------------------------------------------------------
    # 2) Coordinate-compress all item prices (P) because they can be up to 1e9
    #    We will need to quickly "remove" an item with a certain price if it
    #    gets a coupon, so we store them in a segment tree as counts of items.
    # ---------------------------------------------------------
    # Collect all prices in a list for compression
    all_prices = sorted(set(P))
    # Dictionary for price -> compressed index
    # We'll use 0-based indices in the segment tree
    comp_idx = {}
    for i, v in enumerate(all_prices):
        comp_idx[v] = i

    # Build freq array (counts of each price)
    freq = [0]*(len(all_prices))
    for price in P:
        freq[comp_idx[price]] += 1

    # ---------------------------------------------------------
    # 3) Build a segment tree over freq to find and remove an item
    #    with smallest index >= some position that has freq>0.
    #    We want an O(log N) method to "findFirst" such index.
    # ---------------------------------------------------------
    size = len(all_prices)       # length of compressed array
    segsize = 1
    while segsize < size:
        segsize <<= 1

    segtree = [0]*(2*segsize)

    # Build function
    def build():
        # put freq[] into leaves
        for i in range(size):
            segtree[segsize + i] = freq[i]
        # build upward
        for i in range(segsize - 1, 0, -1):
            segtree[i] = segtree[i << 1] + segtree[i << 1 | 1]

    def update(pos, val):
        # pos is 0-based index in freq
        # val is the amount to add (can be negative)
        idx = pos + segsize
        segtree[idx] += val
        idx >>= 1
        while idx > 0:
            segtree[idx] = segtree[idx << 1] + segtree[idx << 1 | 1]
            idx >>= 1

    def get_sum(left, right):
        """
        Returns sum in the interval [left, right], 0-based indexing.
        """
        res = 0
        left += segsize
        right += segsize
        while left <= right:
            if (left & 1) == 1:
                res += segtree[left]
                left += 1
            if (right & 1) == 0:
                res += segtree[right]
                right -= 1
            left >>= 1
            right >>= 1
        return res

    def find_first(pos):
        """
        Returns the smallest index i >= pos such that freq[i] > 0.
        If none exists, returns -1.
        We'll do a segment-tree based search.
        """
        # If total items in [pos, end] is 0, return -1 immediately
        if pos >= size:
            return -1
        if get_sum(pos, size - 1) == 0:
            return -1

        # We'll descend the tree to find the leftmost index >= pos with freq>0
        idx = 1
        left = 0
        right = segsize - 1
        # We'll keep narrowing [left, right] until we get a single leaf
        while left < right:
            mid = (left + right) >> 1
            left_idx = idx << 1
            right_idx = left_idx | 1
            # If pos is in the right half, skip the left half
            if pos <= mid:
                # sum in left child for [pos, mid] if pos>left
                # but we need to handle partial intervals carefully
                # We'll first see if there's anything in the left half
                if get_sum(max(pos, left), mid) > 0:
                    # go left
                    idx = left_idx
                    right = mid
                else:
                    # go right
                    idx = right_idx
                    left = mid+1
            else:
                # pos > mid => skip left side entirely
                idx = right_idx
                left = mid+1
        return left  # or right

    # Build the tree
    build()

    # ---------------------------------------------------------
    # 4) Sort coupons by descending discount D. Then for each coupon,
    #    find an item with price >= L_i (i.e. compressed index >=
    #    the position of L_i in all_prices). If found, use it and reduce total cost.
    # ---------------------------------------------------------
    coupons = [(D[i], L[i]) for i in range(M)]
    coupons.sort(key=lambda x: x[0], reverse=True)  # descending by D

    import bisect

    for discount, limit in coupons:
        # Find the smallest price >= limit
        idx = bisect.bisect_left(all_prices, limit)  # 0-based
        if idx == len(all_prices):
            # no price is >= limit
            continue
        # find the next item that is not used (freq>0) from idx onward
        nxt_pos = find_first(idx)
        if nxt_pos == -1:
            continue
        # use this item
        update(nxt_pos, -1)
        total_cost -= discount

    print(total_cost)

# Call solve() after definition
if __name__ == "__main__":
    solve()