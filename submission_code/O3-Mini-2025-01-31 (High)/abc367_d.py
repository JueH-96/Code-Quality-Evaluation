def main():
    import sys
    
    data = sys.stdin.buffer.read().split()
    if not data:
        return
    it = iter(data)
    N = int(next(it))
    M = int(next(it))
    A = [int(next(it)) for _ in range(N)]
    
    # We want to count the number of ordered pairs (s,t) (with s != t) such that
    # the clockwise distance from rest area s to rest area t is a multiple of M.
    # Number the rest areas 1,2,...,N.
    # Let P[0] = 0 and for i=1..N, let P[i] be the total steps from rest area 1
    # to rest area i+1 (with the wrap-around at N+1 = 1). Then the clockwise distance
    # from s to t (s != t) is:
    #   if s < t:  distance = P[t-1] - P[s-1]
    #   if s > t:  distance = (P[N] - P[s-1]) + P[t-1]
    # We need this distance to be ≡ 0 (mod M).
    #
    # Define S[i] = P[i] mod M for i = 0,..., N-1 (these correspond to rest areas 1,...,N)
    # and let T = P[N] mod M.
    #
    # Then:
    #  • For pairs with s < t (i.e. indices i < j in S), the distance is:
    #         S[j] - S[i]  (mod M)
    #     So the condition becomes S[i] == S[j].
    #
    #  • For pairs with s > t (i.e. indices i > j), the distance is:
    #         T + S[j] - S[i]  (mod M)
    #     So the condition becomes S[i] == S[j] + T  (mod M).
    #
    # We can count these two cases separately.
    
    # Compute the cumulative prefix remainders S and total remainder T.
    S = [0] * N
    cur = 0
    for i in range(N):
        S[i] = cur
        cur = (cur + A[i]) % M
    T = cur  # This is P[N] mod M.
    
    # First part: Count pairs (s,t) with s < t such that S[s-1] == S[t-1]
    freq = {}
    for val in S:
        freq[val] = freq.get(val, 0) + 1
    count_first = 0
    for f in freq.values():
        if f > 1:
            count_first += f * (f - 1) // 2

    # Second part: Count pairs (s,t) with s > t.
    # Here indices in S: for i > j, we need S[i] == (S[j] + T) mod M.
    # We can process the S array in order: for each current index i, count how many previous
    # indices j (with j < i) have S[j] equal to (S[i] - T) mod M.
    count_second = 0
    seen = {}  # dictionary to count frequencies of S[j] for j's we've already seen.
    for x in S:
        needed = (x - T) % M
        count_second += seen.get(needed, 0)
        seen[x] = seen.get(x, 0) + 1

    ans = count_first + count_second
    sys.stdout.write(str(ans))
    
if __name__ == '__main__':
    main()