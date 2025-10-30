def main():
    import sys
    input_data = sys.stdin.read().strip().split()
    N = int(input_data[0])
    S = input_data[1:]  # list of N strings
    
    # We will store all inserted strings in a trie.
    # Each node has:
    #   - a dictionary "children" to map a character to the next node
    #   - a value "min_len" storing the minimum length of any string that goes through this node.
    #
    # To transform T into some already-inserted string U, the cost is:
    #   cost(T -> U) = |T| + |U| - 2 * LCP(T, U).
    # So if we can match T up to depth d in the trie (its LCP with some U is d),
    # and that trie node records min_len = min|U| among all U with that prefix,
    # then cost is |T| + min_len - 2*d.
    #
    # We also allow T to become empty with cost = |T|.
    # We process S in order; for k from 1 to N, we either transform S_k to empty or
    # to one of S_1..S_{k-1}. Then we insert S_k into the trie.

    # Precompute maximum number of trie nodes needed = sum of lengths + 1 (for root).
    max_nodes = sum(len(x) for x in S) + 1
    
    # children[i] will be a dict of {char: next_node_index}
    children = [{} for _ in range(max_nodes)]
    # min_len[i] is the minimal length of a string passing through node i
    min_len = [10**9] * max_nodes
    
    # The root is node 0, and node_count is how many nodes we've used so far
    node_count = 1  # next new node index
    
    def insert(st):
        nonlocal node_count
        node = 0
        n_st = len(st)
        # update min_len at root
        if min_len[node] > n_st:
            min_len[node] = n_st
        for ch in st:
            nxt = children[node].get(ch, -1)
            if nxt == -1:
                nxt = node_count
                children[node][ch] = nxt
                min_len[nxt] = 10**9
                node_count += 1
            node = nxt
            if min_len[node] > n_st:
                min_len[node] = n_st
    
    def query(st):
        # cost to empty is just len(st)
        cost_empty = len(st)
        cost_match = 10**9
        node = 0
        matched_len = 0
        
        # check cost at root
        cost_match = min(cost_match, len(st) + min_len[node] - 2 * matched_len)
        
        for ch in st:
            if ch not in children[node]:
                break
            node = children[node][ch]
            matched_len += 1
            # cost if we match matched_len characters
            cost_match = min(cost_match, len(st) + min_len[node] - 2 * matched_len)
        
        return min(cost_empty, cost_match)
    
    # We'll accumulate answers to print after processing.
    out = []
    
    # Process each string S[k] in order
    for i in range(N):
        if i == 0:
            # No strings in trie yet, so cost = length(S[0])
            ans = len(S[i])
            out.append(ans)
            # Now insert S[0]
            insert(S[i])
        else:
            # Query the minimum cost to match any previously inserted string or empty
            ans = query(S[i])
            out.append(ans)
            # Insert S[i] into trie
            insert(S[i])
    
    print("
".join(map(str, out)))

# Do not forget to call main() at the end
main()