import sys

INF = 10**18

class Node:
    __slots__ = ['min_val', 'children']
    def __init__(self):
        self.min_val = INF
        self.children = {}

def main():
    data = sys.stdin.read().splitlines()
    n = int(data[0])
    strings = []
    for i in range(1, n+1):
        strings.append(data[i].strip())
    
    trie = Node()
    ans_list = []
    
    for s in strings:
        L = len(s)
        candidate = INF
        cur = trie
        candidate = min(candidate, L + cur.min_val)
        
        for char in s:
            if char in cur.children:
                cur = cur.children[char]
                candidate = min(candidate, L + cur.min_val)
            else:
                break
        
        ans_i = min(L, candidate)
        ans_list.append(ans_i)
        
        cur = trie
        if L < cur.min_val:
            cur.min_val = L
        
        depth = 0
        for char in s:
            depth += 1
            if char not in cur.children:
                cur.children[char] = Node()
            cur = cur.children[char]
            value_here = L - 2 * depth
            if value_here < cur.min_val:
                cur.min_val = value_here
                
    for res in ans_list:
        print(res)

if __name__ == "__main__":
    main()