import sys

class Node:
    __slots__ = ['best', 'children']
    def __init__(self):
        self.best = 10**15    # a big number
        self.children = {}

def main():
    data = sys.stdin.read().splitlines()
    n = int(data[0])
    strings = []
    for i in range(1, 1 + n):
        strings.append(data[i].strip())
    
    root = Node()
    results = []
    
    for s in strings:
        L = len(s)
        candidate = root.best
        cur = root
        for c in s:
            if c in cur.children:
                cur = cur.children[c]
                if cur.best < candidate:
                    candidate = cur.best
            else:
                break
        
        ans = min(L, L + candidate)
        results.append(str(ans))
        
        cur = root
        if L < cur.best:
            cur.best = L
        for i, c in enumerate(s):
            if c not in cur.children:
                cur.children[c] = Node()
            cur = cur.children[c]
            depth = i + 1
            val = L - 2 * depth
            if val < cur.best:
                cur.best = val
                
    print("
".join(results))

if __name__ == "__main__":
    main()