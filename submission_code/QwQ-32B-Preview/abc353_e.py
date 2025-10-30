class TrieNode:
    def __init__(self):
        self.children = {}
        self.count = 0

class Trie:
    def __init__(self):
        self.root = TrieNode()
    
    def insert(self, word):
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
            node.count += 1

def sum_of_common_prefix_lengths(N, strings):
    trie = Trie()
    for s in strings:
        trie.insert(s)
    
    total_sum = 0
    queue = [(trie.root, 0)]
    for node, level in queue:
        for child in node.children.values():
            count = child.count
            if count >= 2:
                total_sum += (count * (count - 1)) // 2
            queue.append((child, level + 1))
    return total_sum

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    N = int(data[0])
    strings = data[1:N+1]
    result = sum_of_common_prefix_lengths(N, strings)
    print(result)

if __name__ == "__main__":
    main()