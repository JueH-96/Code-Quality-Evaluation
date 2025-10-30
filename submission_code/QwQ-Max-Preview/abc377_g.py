class TrieNode:
    __slots__ = ['children', 'max_val']
    def __init__(self):
        self.children = {}
        self.max_val = -float('inf')

def main():
    import sys
    input = sys.stdin.read().split()
    n = int(input[0])
    strings = input[1:n+1]
    
    root = TrieNode()
    for k in range(n):
        s = strings[k]
        if k == 0:
            print(len(s))
            current = root
            for a in range(len(s) + 1):
                val = 2 * a - len(s)
                if val > current.max_val:
                    current.max_val = val
                if a < len(s):
                    c = s[a]
                    if c not in current.children:
                        current.children[c] = TrieNode()
                    current = current.children[c]
            continue
        
        current = root
        max_val = current.max_val
        for a in range(1, len(s) + 1):
            c = s[a-1]
            if c not in current.children:
                break
            current = current.children[c]
            if current.max_val > max_val:
                max_val = current.max_val
        
        if max_val == -float('inf'):
            ans = len(s)
        else:
            candidate = len(s) - max_val
            ans = min(len(s), candidate)
        print(ans)
        
        current_insert = root
        for a in range(len(s) + 1):
            val_insert = 2 * a - len(s)
            if val_insert > current_insert.max_val:
                current_insert.max_val = val_insert
            if a < len(s):
                c = s[a]
                if c not in current_insert.children:
                    current_insert.children[c] = TrieNode()
                current_insert = current_insert.children[c]

if __name__ == "__main__":
    main()