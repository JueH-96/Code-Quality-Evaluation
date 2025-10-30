def main():
    import sys
    input_data = sys.stdin.read().strip().split()
    N, K = map(int, input_data[:2])
    lost = set(map(int, input_data[2:]))

    # -----------------------------------------------------------------------
    # 1) Build the multiset of remaining socks (colors).
    #    Each color i has either 2 socks if not lost, or 1 sock if lost.
    #    We then collect these color indices into a list S.
    # -----------------------------------------------------------------------
    counts = [2]*N
    for c in lost:
        counts[c-1] = 1  # lost one sock of this color

    # Build the list of sock-colors
    S = []
    for i, cnt in enumerate(counts):
        color = i+1
        # Append 'color' exactly 'cnt' times
        for _ in range(cnt):
            S.append(color)

    # Sort them
    S.sort()

    L = len(S)  # This is 2N - K

    # -----------------------------------------------------------------------
    # A known result for minimizing ∑|i - j| when pairing is to sort
    # and then form pairs of "adjacent" socks in the sorted order.
    #
    # - If L is even, we simply pair them consecutively: (S[0],S[1]), (S[2],S[3])...
    #   The cost is sum( S[2i+1] - S[2i] ) for i = 0..(L/2 -1).
    #
    # - If L is odd, exactly one sock must remain unpaired ("skipped").
    #   We want to choose which sock to skip so that the cost of pairing the rest
    #   is minimized, again in consecutive order once that sock is removed.
    #
    # An efficient way to do this (in O(L)) after sorting is:
    #   pre[i]  = minimal cost to pair up to (but not including) index i
    #             (0-based) with no skips, possibly leaving 1 leftover if i is odd.
    #   post[i] = minimal cost to pair from index i to the end,
    #             possibly leaving 1 leftover if (end - i) is odd.
    #
    # Then if we decide to skip S[i], the total cost is pre[i] + post[i+1].
    #
    # We'll compute:
    #   - pre array of length L+1 (where pre[i] considers S[:i])
    #   - post array of length L+1 (where post[i] considers S[i:])
    #
    # Finally:
    #   if L even: answer = pre[L]
    #   if L odd : answer = min_{0 <= i < L}( pre[i] + post[i+1] )
    # -----------------------------------------------------------------------

    # -----------------------
    # Build pre array
    # pre[i] = minimum cost to pair S[0..i-1] (i.e. first i socks), 
    #          with no skip, leaving 0 or 1 leftover if i is even/odd.
    # 0-based indexing for S, but pre is sized L+1 so that pre[i] references S up to index i-1.
    # -----------------------
    pre = [0]*(L+1)  
    # pre[0] = 0 by definition
    for i in range(1, L+1):
        if i == 1:
            # One sock leftover, no pairs
            pre[i] = 0
        else:
            if i % 2 == 0:
                # Pair the last two
                # cost = pre[i-2] + (S[i-1] - S[i-2]) 
                pre[i] = pre[i-2] + (S[i-1] - S[i-2])
            else:
                # Leave S[i-1] as leftover, same cost as pre[i-1]
                pre[i] = pre[i-1]

    # -----------------------
    # Build post array
    # post[i] = minimum cost to pair S[i..L-1], possibly with leftover
    # We'll fill it from right to left.
    #
    # We'll make post of length L+1 so that post[L] = 0 (empty segment)
    # -----------------------
    post = [0]*(L+1)
    # post[L] = 0
    # fill from i=L-1 down to 0
    for i in range(L-1, -1, -1):
        length = L - i  # number of socks from i..L-1
        if length == 1:
            # only one sock leftover => cost=0
            post[i] = 0
        else:
            if length % 2 == 0:
                # pair the first two
                # cost = (S[i+1] - S[i]) + post[i+2]
                post[i] = post[i+2] + (S[i+1] - S[i])
            else:
                # leave sock i leftover, cost = post[i+1]
                post[i] = post[i+1]

    if L % 2 == 0:
        # We must use all socks => the cost is pre[L]
        print(pre[L])
    else:
        # We must skip exactly one sock
        # The cost is min_{i=0..L-1} ( pre[i] + post[i+1] )
        ans = None
        for i in range(L):
            cost = pre[i] + post[i+1]
            if ans is None or cost < ans:
                ans = cost
        print(ans)