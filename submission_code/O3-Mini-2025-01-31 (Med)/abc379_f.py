def main():
    import sys
    data = sys.stdin.buffer.read().split()
    if not data: 
        return
    it = iter(data)
    try:
        n = int(next(it))
    except StopIteration:
        return
    q = int(next(it))
    N = n
    H = [0]*(N+1)
    for i in range(1, N+1):
        H[i] = int(next(it))
        
    # Compute nxt array: for each i, nxt[i] = smallest j>i with H[j] > H[i].
    nxt = [0]*(N+1)
    stack = []
    for i in range(N, 0, -1):
        while stack and H[stack[-1]] <= H[i]:
            stack.pop()
        nxt[i] = stack[-1] if stack else 0
        stack.append(i)
    
    # Precompute dist and root.
    # dist[i] = number of steps (jumps) from i to the end of chain (not counting i itself).
    # root[i] = terminal node of the chain starting at i.
    dist = [0]*(N+1)
    root = [0]*(N+1)
    for i in range(N, 0, -1):
        if nxt[i] == 0:
            dist[i] = 0
            root[i] = i
        else:
            dist[i] = dist[nxt[i]] + 1
            root[i] = root[nxt[i]]
            
    # Build binary lifting table for nxt.
    max_power = (N).bit_length()  # enough so that 2^max_power >= N
    lift = [[0]*(N+1) for _ in range(max_power)]
    # Base: lift[0][i] = nxt[i]
    for i in range(1, N+1):
        lift[0][i] = nxt[i]
    for k in range(1, max_power):
        for i in range(1, N+1):
            prev = lift[k-1][i]
            if prev:
                lift[k][i] = lift[k-1][prev]
            else:
                lift[k][i] = 0

    # jump(x, k): jump from node x exactly k times following nxt.
    def jump(x, k):
        cur = x
        bit = 0
        while k:
            if k & 1:
                cur = lift[bit][cur]
                if cur == 0:
                    return 0
            k //= 2
            bit += 1
        return cur

    # reachable(a, target): Returns True if (target) is in the chain starting at a.
    # (Use the fact that if a and target are in same chain then
    #  jump(a, dist[a]-dist[target]) must equal target.)
    def reachable(a, target):
        if a == 0 or target == 0:
            return False
        if root[a] != root[target]:
            return False
        diff = dist[a] - dist[target]
        candidate = jump(a, diff)
        return candidate == target

    # first_common(a, b): finds the first common node in the chains starting at a and b.
    # (Assumes a and b are in the same chain.)
    def first_common(a, b):
        if a == 0 or b == 0: 
            return 0
        if root[a] != root[b]:
            return 0
        # Align the nodes: the one earlier in the chain has larger dist.
        if dist[a] > dist[b]:
            a = jump(a, dist[a]-dist[b])
        elif dist[b] > dist[a]:
            b = jump(b, dist[b]-dist[a])
        if a == b:
            return a
        for k in range(max_power-1, -1, -1):
            if lift[k][a] != lift[k][b]:
                a = lift[k][a]
                b = lift[k][b]
        return nxt[a]  # now nxt[a] == nxt[b] is the first common node.
    
    # Now answer each query.
    # For a query (l, r) (with l < r) we work on the "chains"
    # defined as:
    #    A = chain(l) = [l+1, nxt[l+1], nxt[nxt[l+1]], ...]
    #    B = chain(r) = [r+1, nxt[r+1], ...]
    # Our answer is:
    #   - if r is visible from l (i.e. r is in chain(l)), then answer = (number of nodes in chain starting from r)
    #     which is dist[r] (recall: if we define f(r)=dist[r], then f(r) equals the number of visible buildings east of r).
    #   - Otherwise, if chain(l) and chain(r) merge (first common node m), then answer = (number of nodes in m's chain) = dist[m] + 1.
    #   - Otherwise answer = 0.
    out_lines = []
    for _ in range(q):
        l = int(next(it))
        r = int(next(it))
        # If r is the last building, then there is no building east of r.
        if r >= N or l >= N:
            out_lines.append("0")
            continue
        A = l + 1
        B = r + 1
        if reachable(A, r):
            # Building r is visible from l; common tail is chain starting at r.
            # (Recall: chain(r) “length” = number of jumps from r = dist[r].)
            ans = dist[r]
            out_lines.append(str(ans))
        else:
            if A == 0 or B == 0 or root[A] != root[B]:
                out_lines.append("0")
            else:
                m = first_common(A, B)
                if m == 0:
                    out_lines.append("0")
                else:
                    ans = dist[m] + 1
                    out_lines.append(str(ans))
    sys.stdout.write("
".join(out_lines))
    
if __name__ == '__main__':
    main()