class TrieNode:
    __slots__ = ['children', 'min_val']
    def __init__(self):
        self.children = dict()
        self.min_val = float('inf')

def main():
    import sys
    input = sys.stdin.read().split()
    n = int(input[0])
    strings = input[1:n+1]
    
    root = TrieNode()
    
    for i in range(n):
        s = strings[i]
        if i == 0:
            print(len(s))
        else:
            # Query the trie
            current = root
            min_found = current.min_val
            depth = 0
            for c in s:
                if c in current.children:
                    current = current.children[c]
                    depth += 1
                    if current.min_val < min_found:
                        min_found = current.min_val
                else:
                    break
            candidate = len(s) + min_found
            ans = min(candidate, len(s))
            print(ans)
        # Insert into the trie
        current_ins = root
        depth_ins = 0
        val = len(s) - 2 * depth_ins
        if current_ins.min_val > val:
            current_ins.min_val = val
        for c in s:
            if c not in current_ins.children:
                current_ins.children[c] = TrieNode()
            current_ins = current_ins.children[c]
            depth_ins += 1
            val = len(s) - 2 * depth_ins
            if current_ins.min_val > val:
                current_ins.min_val = val

if __name__ == '__main__':
    main()