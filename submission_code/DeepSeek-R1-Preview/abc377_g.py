import sys

def main():
    import sys
    sys.setrecursionlimit(1 << 25)
    n = int(sys.stdin.readline())
    root = {'children': {}, 'min_val': float('inf')}
    for _ in range(n):
        s = sys.stdin.readline().strip()
        a = len(s)
        current_min = a  # Cost to delete everything
        
        # Step 1: Traverse the trie to collect min_vals along the path
        node = root
        d = 0
        collected = [node['min_val']]
        for c in s:
            if c in node['children']:
                node = node['children'][c]
                d += 1
                collected.append(node['min_val'])
            else:
                break
        
        # Compute the minimal candidate
        for val in collected:
            candidate = val + a
            if candidate < current_min:
                current_min = candidate
        
        # Step 2: Insert the current string into the trie
        node = root
        d = 0
        # Update root's min_val
        if (a - 2 * d) < node['min_val']:
            node['min_val'] = a - 2 * d
        for c in s:
            d += 1
            if c not in node['children']:
                node['children'][c] = {'children': {}, 'min_val': float('inf')}
            node = node['children'][c]
            # Update node's min_val
            if (a - 2 * d) < node['min_val']:
                node['min_val'] = a - 2 * d
        
        print(current_min)

if __name__ == '__main__':
    main()