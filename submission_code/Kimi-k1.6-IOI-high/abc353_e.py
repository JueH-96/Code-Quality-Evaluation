def main():
    import sys
    input = sys.stdin.read().split()
    n = int(input[0])
    strings = input[1:n+1]
    
    # Initialize the root node
    root = {'children': {}, 'count': 0}
    
    # Build the trie
    for s in strings:
        node = root
        for c in s:
            if c not in node['children']:
                node['children'][c] = {'children': {}, 'count': 0}
            node = node['children'][c]
            node['count'] += 1
    
    # Traverse the trie and calculate the sum
    total = 0
    stack = list(root['children'].values())
    while stack:
        node = stack.pop()
        total += node['count'] * (node['count'] - 1) // 2
        stack.extend(node['children'].values())
    
    print(total)

if __name__ == "__main__":
    main()