import sys

class TrieNode:
    __slots__ = ['children', 'max_val']
    def __init__(self):
        self.children = [None] * 26
        self.max_val = -float('inf')

def main():
    input = sys.stdin.read().split()
    N = int(input[0])
    strings = input[1:N+1]

    root = TrieNode()

    for s in strings:
        n = len(s)
        reversed_s = s[::-1]
        current_max = -float('inf')
        current_node = root
        for c in reversed_s:
            idx = ord(c) - ord('a')
            if current_node.children[idx] is None:
                break
            current_node = current_node.children[idx]
            if current_node.max_val > current_max:
                current_max = current_node.max_val
        cost = min(n, n - current_max)
        print(cost)

        # Insert the reversed string into the trie
        current_node = root
        for i in range(len(reversed_s)):
            c = reversed_s[i]
            idx = ord(c) - ord('a')
            if current_node.children[idx] is None:
                current_node.children[idx] = TrieNode()
            current_node = current_node.children[idx]
            current_depth = i + 1
            candidate = 2 * current_depth - n
            if candidate > current_node.max_val:
                current_node.max_val = candidate

if __name__ == "__main__":
    main()