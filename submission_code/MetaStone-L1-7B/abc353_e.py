from collections import deque

class Node:
    def __init__(self):
        self.count = 0
        self.children = {}
        self.depth = 0

def main():
    import sys
    input = sys.stdin.read().split()
    N = int(input[0])
    strings = input[1:N+1]

    root = Node()

    for s in strings:
        current = root
        for c in s:
            if c not in current.children:
                child = Node()
                child.depth = current.depth + 1
                current.children[c] = child
            current = current.children[c]
        current.count += 1

    total = 0
    queue = deque([root])

    while queue:
        node = queue.popleft()
        if node.count >= 2:
            c = (node.count * (node.count - 1)) // 2
            for child in node.children.values():
                c -= (child.count * (child.count - 1)) // 2
            total += c * node.depth
        if node.children:
            for child in node.children.values():
                queue.append(child)

    print(total)

if __name__ == "__main__":
    main()