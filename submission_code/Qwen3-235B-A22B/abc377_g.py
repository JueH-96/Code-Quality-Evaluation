class TrieNode:
    __slots__ = ['children', 'min_val', 'depth']
    def __init__(self, depth):
        self.children = {}
        self.min_val = float('inf')
        self.depth = depth

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    N = int(data[0])
    S = data[1:]

    root = TrieNode(0)

    for i in range(N):
        s = S[i]
        if i == 0:
            print(len(s))
            current = root
            for c in s:
                if c not in current.children:
                    current.children[c] = TrieNode(current.depth + 1)
                current = current.children[c]
                val = len(s) - 2 * current.depth
                if val < current.min_val:
                    current.min_val = val
            continue

        current = root
        path_mins = []
        for c in s:
            if c not in current.children:
                break
            current = current.children[c]
            path_mins.append(current.min_val)
        if path_mins:
            min_candidate = min(path_mins)
        else:
            min_candidate = float('inf')
        cost_candidate = len(s) + min_candidate
        ans = min(len(s), cost_candidate)
        print(ans)

        current = root
        for c in s:
            if c not in current.children:
                current.children[c] = TrieNode(current.depth + 1)
            current = current.children[c]
            val = len(s) - 2 * current.depth
            if val < current.min_val:
                current.min_val = val

if __name__ == "__main__":
    main()