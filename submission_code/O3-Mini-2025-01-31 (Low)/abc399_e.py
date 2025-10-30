def main():
    import sys
    data = sys.stdin.read().split()
    if not data:
        return
    n = int(data[0])
    S = data[1].strip()
    T = data[2].strip()

    # Build a mapping function for each letter from 'a' to 'z'.
    # Initially, each letter maps to itself.
    mapping = {chr(ord('a')+i): chr(ord('a')+i) for i in range(26)}

    # For every position in S and T, enforce the transformation rule.
    # If a letter in S is already mapped to something and we get a conflicting target,
    # then it is impossible.
    for s_char, t_char in zip(S, T):
        if mapping[s_char] == s_char:
            mapping[s_char] = t_char
        elif mapping[s_char] != t_char:
            sys.stdout.write("-1")
            return

    # The allowed operation is "replace every occurrence of x with y".
    # In our mapping, the letters for which mapping[letter] != letter must be changed.
    # The cost for "chains" is simply one operation per edge.
    # However, if there is a cycle, e.g. a -> b, b -> a, then an extra operation
    # is needed (using an intermediate letter) to break the cycle.

    # Count the edges that represent a “non-trivial” mapping.
    edges_count = sum(1 for ch in mapping if mapping[ch] != ch)

    # To count cycles, we build a directed graph (function) on the 26 letters.
    # Since each letter maps to exactly one letter, the cycles are disjoint.
    # We use DFS to detect cycles.
    visited = {chr(ord('a')+i): False for i in range(26)}
    rec_stack = {chr(ord('a')+i): False for i in range(26)}
    cycle_count = 0

    def dfs(u):
        nonlocal cycle_count
        visited[u] = True
        rec_stack[u] = True
        # Only consider an edge if it is a change (i.e. u -> mapping[u] when mapping[u] != u)
        v = mapping[u]
        if v != u:
            if not visited[v]:
                dfs(v)
            elif rec_stack[v]:
                # When v is in the current DFS recursion stack,
                # we found a cycle; count it exactly once.
                cycle_count += 1
        rec_stack[u] = False

    for ch in mapping:
        if not visited[ch]:
            dfs(ch)

    # The minimum number of operations required is the sum of:
    #   (1) the number of transformation edges (each edge will be executed once)
    #   (2) plus 1 extra operation for each cycle (to break the cycle)
    result = edges_count + cycle_count
    sys.stdout.write(str(result))


if __name__ == '__main__':
    main()