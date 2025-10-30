# YOUR CODE HERE
import sys
import threading

sys.setrecursionlimit(1 << 25)

def main():
    import sys
    input = sys.stdin.readline

    N_and_strings = sys.stdin.read().split()
    N = int(N_and_strings[0])
    S = N_and_strings[1:]

    class TrieNode:
        def __init__(self):
            self.children = {}
            self.count_end_here = 0
            self.n_total = 0  # Total number of strings under this node

    root = TrieNode()

    for s in S:
        node = root
        for c in s:
            if c not in node.children:
                node.children[c] = TrieNode()
            node = node.children[c]
        node.count_end_here +=1

    ans = [0]

    def dfs(node, depth):
        n_total = node.count_end_here
        total_pairs_N_i = 0

        total_pairs_children = 0

        for child in node.children.values():
            child_pairs = dfs(child, depth +1)
            n_i = child.n_total
            total_pairs_N_i += child_pairs
            n_total += n_i

        node.n_total = n_total

        total_pairs_N = n_total * (n_total -1) //2
        pairs_differ_here = total_pairs_N - total_pairs_N_i
        ans[0] += pairs_differ_here * depth

        return total_pairs_N

    dfs(root, 0)

    print(ans[0])

threading.Thread(target=main).start()