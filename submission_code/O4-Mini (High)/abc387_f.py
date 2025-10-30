import sys
def main():
    import sys
    sys.setrecursionlimit(10000)
    data = sys.stdin.read().split()
    it = iter(data)
    try:
        N = int(next(it))
    except StopIteration:
        return
    M = int(next(it))
    A = [int(next(it)) - 1 for _ in range(N)]
    mod = 998244353

    # 1) Find all cycles in the functional graph
    visited = [False] * N
    in_stack = [False] * N
    cycles = []
    for i in range(N):
        if visited[i]:
            continue
        u = i
        stack = []
        # follow the chain until we revisit
        while not visited[u]:
            visited[u] = True
            in_stack[u] = True
            stack.append(u)
            u = A[u]
        # if u is still in current stack, we've found a new cycle
        if in_stack[u]:
            cycle = []
            # pop until we hit u
            while True:
                v = stack.pop()
                in_stack[v] = False
                cycle.append(v)
                if v == u:
                    break
            cycle.reverse()
            cycles.append(cycle)
        # clear in_stack for any remaining nodes in this exploration
        for v in stack:
            in_stack[v] = False

    # mark which nodes are in any cycle
    on_cycle = [False] * N
    for cyc in cycles:
        for u in cyc:
            on_cycle[u] = True

    # 2) Build children lists for the tree edges (ignore cycle edges)
    children = [[] for _ in range(N)]
    for i in range(N):
        if not on_cycle[i]:
            p = A[i]
            children[p].append(i)
    ch = children  # local alias

    ans = 1
    M_l = M
    mod_l = mod

    # 3) For each cycle, compute its contribution
    for cyc in cycles:
        # f_cycle[m-1] will accumulate the product of g_c(m) over cycle nodes c
        f_cycle = None

        for root in cyc:
            # a) get post-order of the tree T rooted at 'root'
            post = []
            stack = [(root, 0)]
            while stack:
                u, idx = stack.pop()
                cu = ch[u]
                if idx < len(cu):
                    # push to revisit after child
                    stack.append((u, idx + 1))
                    # then process that child
                    stack.append((cu[idx], 0))
                else:
                    post.append(u)

            # b) DP over post-order to compute g_root
            g_map = {}
            last_f = None

            for u in post:
                cu = ch[u]
                # start f_u as all ones
                f_u = [1] * M_l
                # combine over children
                for v in cu:
                    fv = g_map.pop(v, None)
                    if fv is not None:
                        # prefix-sum fv in-place
                        prev = fv[0]
                        for i in range(1, M_l):
                            s_ = fv[i] + prev
                            if s_ >= mod_l:
                                s_ -= mod_l
                            fv[i] = s_
                            prev = s_
                        # multiply into f_u
                        # alias locals
                        fu = f_u
                        modv = mod_l
                        for i in range(M_l):
                            fu[i] = fu[i] * fv[i] % modv
                last_f = f_u
                # only keep g_map[u] if u has children (to save memory)
                if cu:
                    g_map[u] = f_u

            g_c = last_f  # this is g(root)

            # multiply into cycle's f_cycle
            if f_cycle is None:
                # first cycle node: just adopt g_c array
                f_cycle = g_c
            else:
                fc = f_cycle
                modv = mod_l
                for i in range(M_l):
                    fc[i] = fc[i] * g_c[i] % modv

        # sum over m=1..M of f_cycle[m-1]
        total = sum(f_cycle) % mod_l
        ans = ans * total % mod_l

    # 4) Output
    print(ans)


if __name__ == "__main__":
    main()