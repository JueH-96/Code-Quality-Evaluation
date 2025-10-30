def main():
    import sys
    data = sys.stdin.read().split()
    if not data:
        return
    it = iter(data)
    N = int(next(it))
    M = int(next(it))
    A = [int(next(it)) for _ in range(N)]
    B = [int(next(it)) for _ in range(M)]
    
    # We want to answer, for each sushi with deliciousness b,
    # "what is the smallest index i (from 1 to N) such that A[i] <= b?"
    #
    # A straightforward simulation would take O(N) per query, which is too slow
    # for N, M up to 200,000. Instead, note that the condition is static:
    # Person i always takes any sushi with b >= A[i]. Since they are tried in order,
    # the answer for a given b is simply the minimum i with A[i] <= b.
    #
    # Since A[i] and b are in the range [1, 200000], we can precompute for each value t
    # (1 <= t <= X with X = max(max(A), max(B))) the minimal index i among persons with gourmet level <= t.
    # We do this by first computing, for each specific threshold value, the earliest person (smallest index)
    # that has exactly that threshold. Then we compute a prefix-minimum over these values.
    
    maxA = max(A)
    maxB = max(B)
    X = max(maxA, maxB)
    INF = 10**9  # A sentinel; note N<=200000, so INF is sufficiently high.
    
    # best[t] will record the smallest person index with gourmet level exactly t.
    best = [INF] * (X + 1)
    for i, a in enumerate(A, start=1):
        if a <= X:
            if i < best[a]:
                best[a] = i

    # f[t] will be the smallest index among all persons with threshold <= t.
    # Because as b increases, more persons qualify so the answer can only get (or remain) lower.
    f = [INF] * (X + 1)
    f[0] = INF
    for t in range(1, X + 1):
        f[t] = best[t] if best[t] < f[t - 1] else f[t - 1]
    
    # For each sushi piece with deliciousness b, output the answer.
    # If f[b] is INF, then no person qualifies (output -1), otherwise output f[b].
    out_lines = []
    for b in B:
        out_lines.append(str(f[b]) if f[b] != INF else "-1")
    sys.stdout.write("
".join(out_lines))

if __name__ == '__main__':
    main()