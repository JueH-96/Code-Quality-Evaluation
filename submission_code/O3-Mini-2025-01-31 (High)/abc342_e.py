# YOUR CODE HERE
def main():
    import sys, heapq
    # Read all input tokens (using sys.stdin.buffer for speed)
    data = sys.stdin.buffer.read().split()
    if not data:
        return
    it = iter(data)
    try:
        N = int(next(it))
    except StopIteration:
        return
    M = int(next(it))
    # We choose an INF value that is larger than any possible departure time.
    # (The maximum departure time from a piece is l + d*(k-1) and since l,d,k <= 10^9,
    # this is about 1e18. We choose INF = 10**20 to be safe.)
    INF = 10**20

    # Build a "reverse graph": for each station B, we store all train pieces coming
    # into B. Each train piece is stored as a tuple: (A, l, d, k, c) meaning that
    # from station A there is a piece that gives trains with departure times
    # l, l+d, …, l+d*(k-1) and travel time c to arrive at B.
    rev = [[] for _ in range(N+1)]
    for _ in range(M):
        l = int(next(it))
        d = int(next(it))
        k = int(next(it))
        c = int(next(it))
        A = int(next(it))
        B = int(next(it))
        rev[B].append((A, l, d, k, c))
        
    # f[v] will hold the maximum time that one can be at station v and still eventually
    # reach station N. (For stations that are unreachable, we use 0.)
    f = [0]*(N+1)
    f[N] = INF  # at the destination we set f(N)=INF (so that any train arriving here is ok)

    # use a max–heap (priority queue). We store items as (-f[v], v)
    heap = []
    heapq.heappush(heap, (-f[N], N))

    # Process improvement events
    while heap:
        cur_val, v = heapq.heappop(heap)
        cur_val = -cur_val
        # skip outdated entries
        if f[v] != cur_val:
            continue
        # For every train piece (u -> v) we try to update f[u]
        for u, l, d, k, c in rev[v]:
            # We have f(v)=cur_val.
            # A train from u (with parameters l,d,k,c) can be taken if f(v) >= c + l.
            if cur_val < c + l:
                continue
            # Let X = f(v) so that we must have departure time T from u with T <= f(v) – c.
            # The available departure times from u in that piece are:
            #   l, l+d, l+2d, …, l+d*(k–1)
            # The largest member not exceeding (f(v) – c) is computed as:
            #   T = (f(v) – c) – ((f(v) – c – l) mod d)
            Xp = cur_val - c
            candidate = Xp - ((Xp - l) % d)
            # However, even if f(v) is very big, you cannot exceed the “ceiling”:
            max_departure = l + d*(k - 1)
            if candidate > max_departure:
                candidate = max_departure
            # If candidate is improved then update f[u] and push it to the heap.
            if candidate > f[u]:
                f[u] = candidate
                heapq.heappush(heap, (-candidate, u))

    # Output the answer for stations 1 ... N-1.
    # We output "Unreachable" if f[v] is 0.
    out_lines = []
    for i in range(1, N):
        if f[i] == 0:
            out_lines.append("Unreachable")
        else:
            out_lines.append(str(f[i]))
    sys.stdout.write("
".join(out_lines))
    
if __name__ == "__main__":
    main()  # don't forget to call main() exactly as requested!