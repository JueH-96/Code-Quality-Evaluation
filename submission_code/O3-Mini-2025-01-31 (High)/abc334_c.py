def main():
    import sys
    data = sys.stdin.read().split()
    if not data:
        return
    it = iter(data)
    try:
        N = int(next(it))
        K = int(next(it))
    except StopIteration:
        return

    # The lost sock colors; they are given in increasing order.
    lost_set = set()
    for _ in range(K):
        lost_set.add(int(next(it)))
    
    # Originally, for each color i from 1 to N:
    # • if i is in lost_set then only one sock remains,
    # • otherwise, both socks remain.
    # We build an array S of all remaining socks’ colors. (They are inherently sorted by color.)
    S = []
    for i in range(1, N + 1):
        if i in lost_set:
            S.append(i)
        else:
            S.append(i)
            S.append(i)
    # Total number of socks (remains): n = 2N - K.
    n = len(S)
    
    # When pairing socks, a pair with the same color gives weirdness 0.
    # Notice that any color that contributes two copies (i.e. non‐lost)
    # can be paired internally for free. But the ones with a single sock (lost colors)
    # must be paired with a sock of some other color.
    # However, we are allowed to break a non‐lost pair if that helps us
    # rearrange the ordering so that in the final matching the overall (absolute difference)
    # sum is minimized.
    #
    # It turns out that if you sort the socks by color (which S already is)
    # then the minimum total weirdness for pairing almost all socks (all except one if odd)
    # is achieved by choosing a matching that pairs socks that are “neighbors” in S.
    # (This is a well–known fact for pairing points optimally on a line.)
    #
    # So our plan is:
    #   1. If n is even, we simply pair S[0] with S[1], S[2] with S[3], etc.
    #   2. If n is odd, we must leave exactly one sock unpaired.
    #      We choose an index j (which we will require to be even so that when we remove S[j]
    #      the two remaining pieces (prefix and suffix) have even lengths)
    #      and compute the sum of pairing costs by pairing adjacent socks.
    
    # Case 1: n is even.
    if n % 2 == 0:
        ans = 0
        for i in range(0, n, 2):
            ans += S[i+1] - S[i]
        sys.stdout.write(str(ans))
        return
    
    # Case 2: n is odd.
    # Precompute prefix pairing cost.
    # For any even j between 0 and n, let P[j] be the cost obtained by pairing S[0:j]
    # as (S[0],S[1]), (S[2],S[3]), ..., (S[j-2], S[j-1]).
    P = [0] * (n + 1)
    P[0] = 0
    # We only fill P for even indices.
    for j in range(2, n, 2):
        P[j] = P[j-2] + (S[j-1] - S[j-2])
    
    # Precompute suffix pairing cost.
    # For an odd starting index i (this will occur when i = j+1, where j is even),
    # let R[i] be the pairing cost for S[i:n] where (n - i) is even.
    R = [0] * (n + 1)
    R[n] = 0  # S[n:n] is empty.
    # We fill R for odd indices only.
    # Since n is odd, j (which is the removed sock index) will be even so j+1 is odd.
    # For i in {n-2, n-4, …} that are odd:
    for i in range(n - 2, 0, -2):  
        R[i] = (S[i+1] - S[i]) + (R[i+2] if (i+2) <= n else 0)
    
    # Now, we try all candidates for the sock to leave unpaired.
    # We require to remove S[j] with j even so that both S[0:j] and S[j+1:n] have even lengths.
    best = None
    for j in range(0, n, 2):
        # Total cost = cost for prefix S[0:j] plus cost for suffix S[j+1:n].
        cost = P[j] + (R[j+1] if (j+1) <= n else 0)
        if best is None or cost < best:
            best = cost
    sys.stdout.write(str(best))
    
if __name__ == '__main__':
    main()