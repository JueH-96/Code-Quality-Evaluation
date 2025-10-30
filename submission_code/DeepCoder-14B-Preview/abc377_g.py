def main():
    import sys
    input = sys.stdin.read().split()
    n = int(input[0])
    strings = input[1:n+1]
    
    # Initialize the trie
    root = {
        'children': {},
        'min_val': float('inf'),  # This value is never used
    }
    
    for s in strings:
        current_min = len(s)
        current_node = root
        # Process the string to find minimal cost
        for i in range(len(s)):
            char = s[i]
            if char not in current_node['children']:
                break
            current_node = current_node['children'][char]
            if 'min_val' in current_node and current_node['min_val'] != float('inf'):
                option = len(s) + current_node['min_val']
                if option < current_min:
                    current_min = option
        print(current_min)
        # Insert the string into the trie
        current_node = root
        for i in range(len(s)):
            char = s[i]
            if char not in current_node['children']:
                current_node['children'][char] = {
                    'children': {},
                    'min_val': float('inf'),
                }
            current_node = current_node['children'][char]
            current_val = len(s) - 2 * (i + 1)
            if current_val < current_node['min_val']:
                current_node['min_val'] = current_val

if __name__ == '__main__':
    main()