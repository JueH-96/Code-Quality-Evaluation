def main():
    import sys
    sys.setrecursionlimit(10**7)
    input_data = sys.stdin.read().strip().split()
    N = int(input_data[0])
    S = input_data[1:]
    
    # We will build a trie incrementally. For each k from 1..N:
    # - We compute the cost to transform T = S[k-1] to either empty or one of {S[0],...,S[k-2]}.
    # - Then insert S[k-1] into the trie for potential use by future strings.
    #
    # Cost to transform T into an existing string X is:
    #   cost = (|T| - LCP) + (|X| - LCP) = |T| + |X| - 2 * LCP
    # We compare also with cost to transform T into empty = |T|.
    #
    # In the trie, each node will store:
    #   - children: dict of char -> child_id
    #   - min_len_subtree: the minimum length of any string that has this node as a prefix
    #
    # To query the cost for T:
    #   - Walk T character by character in the trie.
    #   - At step i (0-based), if we can go to child c = T[i], we do so, let depth = i+1
    #     then possible cost if we match depth = (|T| + min_len_subtree of current node - 2 * depth).
    #   - Keep track of the minimum such cost while we can follow T in the trie.
    #   - Compare that with cost to empty = |T|.
    #
    # Then we insert T into the trie, updating min_len_subtree along the path.

    # We'll implement a list-based node storage. Each node is:
    #   children (dict), min_len_subtree (int).
    # Because total length sum is up to 2*10^5, we won't exceed that many total nodes.

    sys.setrecursionlimit(10**7)

    # Each node: children={}, min_len_subtree=large
    # We'll keep them in arrays for speed:
    #   children[node] = dict()  (mapping char -> next_node)
    #   min_len[node] = integer
    # We'll have a global "trie" with root = 0.

    children = []
    min_len_subtree = []

    def new_node():
        children.append({})
        min_len_subtree.append(10**10)
        return len(children) - 1

    root = new_node()  # root node

    def insert_string(st):
        # Insert st into trie, update min_len_subtree
        curr = root
        length = len(st)
        min_len_subtree[curr] = min(min_len_subtree[curr], length)
        for c in st:
            if c not in children[curr]:
                nxt = new_node()
                children[curr][c] = nxt
            curr = children[curr][c]
            min_len_subtree[curr] = min(min_len_subtree[curr], length)

    def query_cost(st):
        # Returns the minimum cost to transform st into
        # either empty or one of the strings in trie
        # (the set of strings inserted so far).
        # cost_empty = len(st)
        # cost_match = min over any prefix match: len(st) + min_len_subtree - 2*depth
        n = len(st)
        ans = n  # cost to make st empty
        curr = root
        depth = 0
        if len(children[curr]) == 0:
            # No strings in trie yet, best is empty
            return ans
        for c in st:
            if c not in children[curr]:
                break
            curr = children[curr][c]
            depth += 1
            # cost if we match depth
            cost_here = n + min_len_subtree[curr] - 2 * depth
            if cost_here < ans:
                ans = cost_here
        return ans

    # We iterate over k in [1..N]:
    # For S_k, we first compute cost, then insert S_k.
    # For k=1, the set is empty, so cost is just |S_1|.
    #
    # We'll collect answers in a list and print at the end.

    out = []
    for i in range(N):
        # Compute cost
        T = S[i]
        cost_k = query_cost(T)
        out.append(str(cost_k))
        # Then insert T
        insert_string(T)

    print('
'.join(out))


# Call main() as required
if __name__ == "__main__":
    main()