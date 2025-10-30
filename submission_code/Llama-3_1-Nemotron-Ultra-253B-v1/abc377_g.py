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
    for i in range(n):
        s = strings[i]
        if i == 0:
            print(len(s))
            M = len(s)
            node = root
            val = -M
            if val > node.max_val:
                node.max_val = val
            for idx, c in enumerate(s):
                if c not in node.children:
                    node.children[c] = TrieNode()
                node = node.children[c]
                L = idx + 1
                val = 2 * L - M
                if val > node.max_val:
                    node.max_val = val
        else:
            current_max = root.max_val
            node = root
            for c in s:
                if c in node.children:
                    node = node.children[c]
                    if node.max_val > current_max:
                        current_max = node.max_val
                else:
                    break
            ans = min(len(s), len(s) - current_max)
            print(ans)
            M = len(s)
            node = root
            val = -M
            if val > node.max_val:
                node.max_val = val
            for idx, c in enumerate(s):
                if c not in node.children:
                    node.children[c] = TrieNode()
                node = node.children[c]
                L = idx + 1
                val = 2 * L - M
                if val > node.max_val:
                    node.max_val = val

if __name__ == '__main__':
    main()