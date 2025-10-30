class TrieNode:
    def __init__(self):
        self.children = {}
        self.min_length = float('inf')

def insert(node, s, length):
    current = node
    for char in s:
        if char not in current.children:
            current.children[char] = TrieNode()
        current = current.children[char]
        current.min_length = min(current.min_length, length)

def query(node, s, M):
    current = node
    P = 0
    min_cost = M
    for char in s:
        if char not in current.children:
            break
        current = current.children[char]
        P += 1
        if current.min_length != float('inf'):
            cost = M + current.min_length - 2 * P
            min_cost = min(min_cost, cost)
    return min_cost

def main():
    import sys
    data = sys.stdin.read().split()
    N = int(data[0])
    strings = data[1:N+1]
    root = TrieNode()
    for Sk in strings:
        M = len(Sk)
        reverse_Tk = Sk[::-1]
        cost = query(root, reverse_Tk, M)
        print(cost)
        insert(root, reverse_Tk, M)

if __name__ == "__main__":
    main()