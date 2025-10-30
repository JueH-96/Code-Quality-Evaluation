def main():
    import sys
    sys.setrecursionlimit(10**7)
    input = sys.stdin.readline
    
    # Read N and Q
    N, Q = map(int, input().split())
    H = list(map(int, input().split()))
    
    # 1) Compute T_left(i) for i=1..N:
    #    T_left(i) = index of the nearest building j < i with H_j > H_i, or 0 if none.
    #    We'll do this with a monotonic stack.
    T_left = [0]*(N+1)  # 1-based indexing; T_left[0] unused
    stack = []
    for i in range(1, N+1):
        # Pop while top of stack has height <= H[i-1] (since H is 0-based).
        # We want strictly greater, so pop if H[stack[-1]-1] <= H[i-1].
        while stack and H[stack[-1]-1] <= H[i-1]:
            stack.pop()
        T_left[i] = stack[-1] if stack else 0
        stack.append(i)
    
    # 2) Build array P of pairs (T_left(i), i) for i=1..N, then sort by T_left(i).
    P = [(T_left[i], i) for i in range(1, N+1)]
    P.sort(key=lambda x: x[0])
    
    # 3) Read queries. We will store them as (l, r, idx) so we can output in correct order.
    queries = []
    for idx in range(Q):
        l, r = map(int, input().split())
        queries.append((l, r, idx))
    
    # 4) Sort the queries by l in ascending order.
    queries.sort(key=lambda x: x[0])
    
    # Fenwick (BIT) for counting indices. We'll store up to N.
    fenwicksum = [0]*(N+1)
    
    def fenw_update(pos, val):
        while pos <= N:
            fenwicksum[pos] += val
            pos += pos & -pos
    
    def fenw_query(pos):
        s = 0
        while pos > 0:
            s += fenwicksum[pos]
            pos -= pos & -pos
        return s
    
    # 5) We'll sweep through P in ascending order of B = T_left(i).
    #    For a query with parameter l, we want to activate all buildings b with T_left(b) < l.
    #    Then the answer is count of those active in [r+1..N].
    answers = [0]*Q
    p = 0  # pointer for array P
    lenP = len(P)
    
    for l, r, qidx in queries:
        # Activate all buildings whose T_left-value < l
        while p < lenP and P[p][0] < l:
            # P[p] = (B, idx)
            _, bidx = P[p]
            fenw_update(bidx, 1)
            p += 1
        # Now we count how many active indices b are in [r+1..N]
        if r < N:
            ans = fenw_query(N) - fenw_query(r)
        else:
            ans = 0
        answers[qidx] = ans
    
    # 6) Output the answers in the original order
    print("
".join(map(str, answers)))

# Don't forget to call main()
if __name__ == "__main__":
    main()