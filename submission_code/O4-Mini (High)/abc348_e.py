import sys
def main():
    import sys
    input = sys.stdin.readline

    n_line = input()
    if not n_line:
        return
    n = int(n_line)

    adj = [[] for _ in range(n)]
    for _ in range(n - 1):
        a, b = map(int, input().split())
        a -= 1
        b -= 1
        adj[a].append(b)
        adj[b].append(a)

    C = list(map(int, input().split()))

    # First DFS (iterative) to compute:
    # 1) sumC[u] = sum of C[i] over subtree of u (rooted at 0)
    # 2) dp1 = sum of C[i] * depth(i) over all i, using root = 0
    sumC = [0] * n
    dp1 = 0
    # stack entries: (node, parent, visited_flag, depth)
    stack = [(0, -1, False, 0)]
    while stack:
        u, p, visited, depth = stack.pop()
        if not visited:
            # Push back as visited, then push children
            stack.append((u, p, True, depth))
            for v in adj[u]:
                if v == p:
                    continue
                stack.append((v, u, False, depth + 1))
        else:
            # Post-order: all children have sumC[v] computed
            s = C[u]
            for v in adj[u]:
                if v == p:
                    continue
                s += sumC[v]
            sumC[u] = s
            dp1 += C[u] * depth

    totalC = sumC[0]

    # Second DFS (iterative) to reroot DP and find minimum f(v)
    ans = dp1
    # stack2 entries: (node, parent, dp_value_for_node)
    stack2 = [(0, -1, dp1)]
    while stack2:
        u, p, dp_u = stack2.pop()
        if dp_u < ans:
            ans = dp_u
        for v in adj[u]:
            if v == p:
                continue
            # Reroot formula: moving root from u to v
            dp_v = dp_u + totalC - 2 * sumC[v]
            stack2.append((v, u, dp_v))

    print(ans)

# Call main to execute
main()