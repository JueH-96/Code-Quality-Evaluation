MOD = 998244353

def main():
    import sys
    sys.setrecursionlimit(10**6)
    data = sys.stdin.read().strip().split()
    if not data:
        return
    it = iter(data)
    n = int(next(it))
    
    # Read the (n-1) matches.
    # Each match is given as: p_i q_i, which means that just before the match, these two players’ teams (which are
    # disjoint) will compete. The team containing p_i (call it team A) has size A; the team containing q_i (team B) has size B.
    # In the match the winning probabilities are:
    #   team A wins with probability A/(A+B)   and   team B wins with probability B/(A+B)
    # And the winning team’s win count increases by 1 for every member of that team.
    #
    # By linearity of expectation, the overall expected win for a player turns out to be exactly the sum of contributions
    # along the unique path from that player's “leaf” (as an initial one‐person team) to the final team in the merge tree.
    # At every match merging two teams (with sizes A and B), if the team that contains a particular player is chosen, then
    # that player gets an additional win with probability equal to the team’s size divided by (A+B). (Notice that this value
    # is deterministic because the sizes are predetermined by the merge order.)
    #
    # Our idea is to build the “merge tree” from the tournament. The leaves represent the original players 1..n. Internal
    # nodes represent merge matches. In match i (with given p_i and q_i), if the team containing p_i has size A and that 
    # of q_i has size B then:
    #   • every player originally in the team containing p_i gets an addition of A/(A+B)
    #   • every player originally in the team containing q_i gets an addition of B/(A+B)
    #
    # We will build the tree as follows:
    #   • Leaves: nodes 1..n – each corresponding to a single player.
    #   • Internal nodes: assigned ids from n+1 to 2n-1 in the order in which the matches happen.
    #
    # To build the tree, we simulate a union‐find where each component is represented by the node id corresponding to the
    # (possibly merged) team. When processing a match (p, q), use find to get the current representative node for p
    # and for q. Let their sizes be A and B respectively. Then create a new node (with new id) with left child = rep_p 
    # and right child = rep_q. Also record the “contribution values” associated to the children:
    #    contr_left = A/(A+B)  and  contr_right = B/(A+B), computed mod MOD.
    # Then set the new node’s size to A+B and update the union‐find to make this new node the representative.
    #
    # Finally, do a DFS from the final (unique) node (the root) to each leaf; as you traverse an edge, add the 
    # corresponding contribution (mod MOD) to the accumulated sum along the path.
    # The accumulated value obtained at a leaf is exactly the expected number of wins for that original player (mod MOD).
    
    matches = []
    for _ in range(n-1):
        p = int(next(it))
        q = int(next(it))
        matches.append((p, q))
    
    total_nodes = n + n - 1  # n leaves and n-1 internal nodes
    # For nodes 1..total_nodes, we store:
    #   left_child[node]  and  right_child[node]: the children (None if leaf)
    #   add_left[node]  and  add_right[node]: the contribution from that edge
    left_child = [None] * (total_nodes + 1)
    right_child = [None] * (total_nodes + 1)
    add_left = [0] * (total_nodes + 1)
    add_right = [0] * (total_nodes + 1)
    # size_arr[node]: the number of original players in that subtree.
    size_arr = [0] * (total_nodes + 1)
    for i in range(1, n+1):
        size_arr[i] = 1  # each individual player's team has size 1 initially.
    
    # Union–find arrays. Initially, each node is its own parent.
    parent = list(range(total_nodes + 1))
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    
    # Function for computing the modular inverse (MOD is prime).
    def modinv(x):
        return pow(x, MOD-2, MOD)
    
    # Next new node id for an internal node.
    next_node = n + 1
    
    # Process the matches in the given order.
    for (p, q) in matches:
        rep_p = find(p)
        rep_q = find(q)
        size_p = size_arr[rep_p]
        size_q = size_arr[rep_q]
        tot = size_p + size_q
        inv_tot = modinv(tot)
        # Compute contributions in mod arithmetic:
        contr_p = (size_p * inv_tot) % MOD
        contr_q = (size_q * inv_tot) % MOD

        # Create a new node to represent the merged team.
        cur = next_node
        next_node += 1
        left_child[cur] = rep_p
        right_child[cur] = rep_q
        add_left[cur] = contr_p
        add_right[cur] = contr_q
        size_arr[cur] = tot

        # Merge the two components in union–find.
        parent[rep_p] = cur
        parent[rep_q] = cur
        parent[cur] = cur  # new node is its own parent
    
    # The final tree’s root is the representative of any original leaf.
    root = find(1)
    
    # Do a DFS over this tree, propagating the accumulated contribution sum.
    # For a leaf (node id 1..n) the accumulated sum is the expected win for that player.
    res = [0] * (n + 1)
    def dfs(node, cur_sum):
        # If leaf, record the accumulated value.
        if left_child[node] is None and right_child[node] is None:
            res[node] = cur_sum % MOD
        else:
            if left_child[node] is not None:
                dfs(left_child[node], (cur_sum + add_left[node]) % MOD)
            if right_child[node] is not None:
                dfs(right_child[node], (cur_sum + add_right[node]) % MOD)
    
    dfs(root, 0)
    
    # Print the answer for players 1 through n separated by spaces.
    out = " ".join(str(res[i] % MOD) for i in range(1, n+1))
    sys.stdout.write(out)

if __name__ == '__main__':
    main()