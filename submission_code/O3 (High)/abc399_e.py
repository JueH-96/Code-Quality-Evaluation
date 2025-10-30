import sys

def main() -> None:
    input = sys.stdin.readline
    n = int(input().strip())
    s = input().strip()
    t = input().strip()

    # mapping[x] = y   (indexes 0‥25).  -1 : x does not appear in S.
    mapping = [-1] * 26
    for sc, tc in zip(s, t):
        si = ord(sc) - 97
        ti = ord(tc) - 97
        if mapping[si] == -1:
            mapping[si] = ti
        elif mapping[si] != ti:          # inconsistent requirement
            print(-1)
            return

    # Letters that actually occur in S
    letters_in_s = set(ord(ch) - 97 for ch in s)

    # Edges that really change a letter
    nodes_with_edge = set()
    operations_for_edges = 0
    for v in letters_in_s:
        if mapping[v] == -1:             # letter never appears in mapping? (shouldn't happen)
            mapping[v] = v
        if mapping[v] != v:              # non-identity mapping, one mandatory operation
            operations_for_edges += 1
            nodes_with_edge.add(v)

    # If nothing has to be changed
    if operations_for_edges == 0:
        print(0)
        return

    # Detect cycles among nodes_with_edge.
    # The function graph is defined only on letters, out-degree ≤ 1.
    visited = [0] * 26          # 0: unvisited, 1: visiting, 2: finished
    extra_ops_for_cycles = 0
    node_set = nodes_with_edge

    for start in list(node_set):
        if visited[start]:
            continue

        stack = []
        u = start
        while True:
            if not visited[u]:
                visited[u] = 1
                stack.append(u)

                v = mapping[u]
                # Edge ends if target letter is not in node_set
                if v not in node_set:
                    for node in stack:
                        visited[node] = 2
                    break
                u = v
                continue

            if visited[u] == 1:          # found a cycle
                # nodes belonging to the cycle are from first appearance of `u` to end of stack
                first = stack.index(u)
                if len(stack) - first > 1:   # length>1 real cycle
                    extra_ops_for_cycles += 1
                # mark all stacked nodes finished
                for node in stack:
                    visited[node] = 2
                break

            # visited[u] == 2  (reached an already processed path)
            for node in stack:
                visited[node] = 2
            break

    # If every letter appears in S and at least one cycle exists, we have no spare letter
    if extra_ops_for_cycles > 0 and len(letters_in_s) == 26:
        print(-1)
        return

    print(operations_for_edges + extra_ops_for_cycles)


if __name__ == "__main__":
    main()