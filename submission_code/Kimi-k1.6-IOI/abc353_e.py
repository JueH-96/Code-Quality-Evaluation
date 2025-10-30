import sys
from collections import deque

def main():
    input = sys.stdin.read().split()
    n = int(input[0])
    strings = input[1:n+1]
    
    # Build the trie
    root = {'count': 0, 'children': {}}
    for s in strings:
        node = root
        for c in s:
            if c not in node['children']:
                node['children'][c] = {'count': 0, 'children': {}}
            node = node['children'][c]
            node['count'] += 1
    
    # Calculate the sum using BFS
    total = 0
    q = deque()
    for child in root['children'].values():
        q.append(child)
    
    while q:
        curr = q.popleft()
        total += curr['count'] * (curr['count'] - 1) // 2
        for child in curr['children'].values():
            q.append(child)
    
    print(total)

if __name__ == "__main__":
    main()