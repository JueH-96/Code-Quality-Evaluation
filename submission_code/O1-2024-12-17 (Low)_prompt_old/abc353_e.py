def solve():
    import sys
    input_data = sys.stdin.read().strip().split()
    N = int(input_data[0])
    strings = input_data[1:]

    # To avoid potential recursion issues with large N
    sys.setrecursionlimit(10**7)

    # We'll implement a Trie using a list of dictionaries for children
    # and a parallel list for counts.
    # Each node has:
    #   - A dictionary of children: char -> node_index
    #   - An integer count of how many strings pass through this node.
    # We start with just a root node (index 0).
    trie_children = [{}]
    trie_count = [0]

    def new_node():
        trie_children.append({})
        trie_count.append(0)
        return len(trie_children) - 1

    # Insert each string into the Trie
    for s in strings:
        node = 0
        trie_count[node] += 1  # root gets +1 for each string
        for ch in s:
            if ch not in trie_children[node]:
                nxt = new_node()
                trie_children[node][ch] = nxt
            node = trie_children[node][ch]
            trie_count[node] += 1

    from collections import deque

    # We want to compute the sum of f(S_i, S_j) over all pairs.
    # The key fact: if count[node] = c at a node of depth d (d >= 1),
    # then every pair of those c strings contributes +1 to their LCP
    # for that depth. Summing over all depths (all non-root nodes),
    # we add (c choose 2) each time.
    #
    # We'll do a BFS or DFS over the Trie (excluding the root's own contribution).
    # Root is at index 0, whose depth is 0. We only add from children downward.
    queue = deque([0])
    ans = 0
    # We do not add choose2 for the root (depth=0). Only for deeper nodes.
    while queue:
        node = queue.popleft()
        for child in trie_children[node].values():
            # Add choose2(count[child]) for each child
            c = trie_count[child]
            ans += c * (c - 1) // 2  # choose2
            queue.append(child)

    print(ans)

def _test():
    # Sample tests
    input_data = """3
ab
abc
arc
"""
    print(run_io_fun(input_data, solve))

    input_data = """11
ab
bb
aaa
bba
baba
babb
aaaba
aabbb
a
a
b
"""
    print(run_io_fun(input_data, solve))

def run_io_fun(input_data, func):
    import sys
    from io import StringIO
    backup_stdin = sys.stdin
    backup_stdout = sys.stdout
    try:
        sys.stdin = StringIO(input_data)
        output_buffer = StringIO()
        sys.stdout = output_buffer
        func()
    finally:
        sys.stdin = backup_stdin
        sys.stdout = backup_stdout
    return output_buffer.getvalue()

# solve()  # Uncomment to run with actual input from stdin