import sys
import threading

def main():
    import sys
    sys.setrecursionlimit(1000000)
    input_data = sys.stdin.read().split()
    n = int(input_data[0])
    strings = input_data[1:]

    # Build a trie. Each node has a dict of children and a count of strings ending here.
    children = []      # List of dicts: children[node] = {char: child_node_index, ...}
    term_count = []    # term_count[node] = number of strings that end exactly at this node
    children.append({})  # root = 0
    term_count.append(0)

    # Insert all strings
    for s in strings:
        node = 0
        for ch in s:
            d = children[node]
            if ch not in d:
                d[ch] = len(children)
                children.append({})
                term_count.append(0)
            node = d[ch]
        term_count[node] += 1

    # Now do a post-order traversal to compute subtree sizes and accumulate answer.
    total = 0
    def dfs(u):
        nonlocal total
        # Start with the number of strings ending exactly at u
        sz = term_count[u]
        # Recurse into children
        for v in children[u].values():
            sz_child = dfs(v)
            sz += sz_child
        # For every node except the root, add C(sz,2)
        if u != 0:
            total += sz * (sz - 1) // 2
        return sz

    dfs(0)
    print(total)

if __name__ == "__main__":
    main()