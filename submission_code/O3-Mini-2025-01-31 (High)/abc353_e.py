def main():
    import sys
    data = sys.stdin.read().split()
    if not data:
        return
    # First value is N, followed by the strings.
    N = int(data[0])
    strings = data[1:]
    
    # We will build a trie where each node contains:
    # - A dictionary mapping characters to child node indices.
    # - A count of how many strings pass through that node.
    # Note:
    #   The root (index 0) represents the empty prefix, which we will ignore in the final sum.
    trie_children = [{}]
    trie_count = [0]
    
    # Build the trie: for each string, traverse letter by letter.
    for s in strings:
        current = 0
        for ch in s:
            # Create a new child if needed.
            if ch not in trie_children[current]:
                trie_children[current][ch] = len(trie_children)
                trie_children.append({})
                trie_count.append(0)
            current = trie_children[current][ch]
            trie_count[current] += 1
    
    # The function f(x, y) for strings is the length of their common prefix.
    # Notice that if two strings share a prefix corresponding to a node at depth d,
    # they contribute 1 for that common letter. Hence, the total sum over all pairs is
    # the sum of "number of pairs having that node in common".
    # For any node (other than the root), if c strings pass through it, then it adds
    # c*(c-1)//2 to the answer.
    ans = 0
    for i in range(1, len(trie_children)):
        c = trie_count[i]
        if c > 1:
            ans += c * (c - 1) // 2
    
    sys.stdout.write(str(ans))

if __name__ == '__main__':
    main()