def main():
    import sys
    from collections import deque
    data = sys.stdin.buffer.read().split()
    if not data:
        return
    it = iter(data)
    # Read number of people and number of insurances
    N = int(next(it))
    M = int(next(it))
    
    # Build the tree structure.
    # For i from 2 to N, person i’s parent is given.
    children = [[] for _ in range(N+1)]
    for i in range(2, N+1):
        p = int(next(it))
        children[p].append(i)
    
    # For each person, we want to record the best (largest) insurance parameter y they purchased.
    # A person who buys insurance with parameter y covers themselves and any descendant v 
    # if distance d(v) - d(u) <= y (where d(u) is the depth of u).
    # We store ins[x] = max_y among all insurances bought by person x, or -1 if none.
    ins = [-1] * (N+1)
    for _ in range(M):
        x = int(next(it))
        y = int(next(it))
        if y > ins[x]:
            ins[x] = y

    # We will compute:
    #  - depth[u]: the depth of person u in the tree (with person 1 at depth 0).
    #  - dp[u]: the maximum value of (d(u') + y) along the ancestor path from the root to u.
    #      (We define f(u') = d(u') + y for an insurance event at u'. If u' didn’t buy insurance, 
    #       set f(u') = -1.)
    # A node v is covered by some insurance if there exists an ancestor u (possibly v itself)
    # such that f(u) >= depth[v]. This is equivalent to dp[v] >= depth[v].
    
    depth = [0] * (N+1)
    dp = [-1] * (N+1)
    # For the root, depth[1] is 0.
    if ins[1] == -1:
        dp[1] = -1
    else:
        dp[1] = 0 + ins[1]  # f(1) = depth[1] + ins[1]
    
    ans = 0
    if dp[1] >= depth[1]:
        ans += 1
    
    # Process the tree in a BFS order:
    dq = deque()
    dq.append(1)
    while dq:
        u = dq.popleft()
        for v in children[u]:
            depth[v] = depth[u] + 1
            # Compute f(v) if person v purchased an insurance.
            if ins[v] == -1:
                f_v = -1
            else:
                f_v = depth[v] + ins[v]
            # dp[v] is the maximum over the path from the root to v.
            # In other words, we update dp[v] = max(dp[u], f(v)).
            dp[v] = dp[u] if dp[u] >= f_v else f_v
            if dp[v] >= depth[v]:
                ans += 1
            dq.append(v)
    sys.stdout.write(str(ans))

if __name__ == '__main__':
    main()