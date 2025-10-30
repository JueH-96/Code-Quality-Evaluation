def main():
    import sys
    data = sys.stdin.read().strip().split()
    N = int(data[0])
    strings = data[1:]
    
    # Trie storage: each element in 'children' is a dict mapping char -> child_node_index
    children = [{}]
    # pass_count[node] = number of strings that pass through this node
    pass_count = [0]
    
    def add_string(s):
        node = 0
        pass_count[node] += 1
        for char in s:
            if char not in children[node]:
                children[node][char] = len(children)
                children.append({})
                pass_count.append(0)
            node = children[node][char]
            pass_count[node] += 1
    
    # Build the trie with all strings
    for s in strings:
        add_string(s)
    
    # Calculate the result:
    # Sum of LCP for all pairs = sum_{node != root} [pass_count[node] choose 2]
    # Because each node at depth d is part of the prefix for pass_count[node] strings
    # => that node contributes 1 to each pair's LCP among those pass_count[node] strings
    # => total contribution for that node = C(pass_count[node], 2)
    ans = 0
    for i in range(1, len(pass_count)):  # skip root (i=0)
        k = pass_count[i]
        ans += k * (k - 1) // 2
    
    print(ans)

# Do not forget to call main().
if __name__ == "__main__":
    main()