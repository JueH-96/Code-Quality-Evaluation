class TrieNode:
    __slots__ = ['children', 'count']
    def __init__(self):
        self.children = {}
        self.count = 0

n = int(input())
strings = input().split()

root = TrieNode()
total = 0

for s in strings:
    current = root
    temp_sum = 0
    for c in s:
        if c in current.children:
            temp_sum += current.children[c].count
            current = current.children[c]
        else:
            break
    total += temp_sum

    current = root
    for c in s:
        if c not in current.children:
            current.children[c] = TrieNode()
        current = current.children[c]
        current.count += 1

print(total)