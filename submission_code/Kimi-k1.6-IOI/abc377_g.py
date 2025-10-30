import sys

def main():
    input = sys.stdin.read().split()
    n = int(input[0])
    strings = input[1:n+1]
    
    root = {'children': {}, 'min_val': float('inf')}
    for s in strings:
        current_node = root
        min_cost = len(s)
        # Query phase
        for m in range(1, len(s) + 1):
            c = s[m-1]
            if c not in current_node['children']:
                break
            current_node = current_node['children'][c]
            candidate = len(s) + current_node['min_val']
            if candidate < min_cost:
                min_cost = candidate
        print(min_cost)
        # Insert phase
        current_node = root
        L = len(s)
        for m in range(1, L + 1):
            c = s[m-1]
            if c not in current_node['children']:
                current_node['children'][c] = {'children': {}, 'min_val': float('inf')}
            current_node = current_node['children'][c]
            val = L - 2 * m
            if val < current_node['min_val']:
                current_node['min_val'] = val

if __name__ == "__main__":
    main()