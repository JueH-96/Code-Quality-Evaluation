# YOUR CODE HERE
import sys
import threading
import math

MOD = 998244353

sys.setrecursionlimit(1 << 25)

def main():
    import sys
    import math
    N_and_rest = sys.stdin.read().split()
    N = int(N_and_rest[0])
    A_list = [int(x) for x in N_and_rest[1:]]
    A_list.insert(0, 0) # 1-based indexing

    intervals = []
    for i in range(1, N+1):
        A_i = A_list[i]
        intervals.append((A_i + 1, i, i)) # (start, end, index)

    intervals.sort(key=lambda x: (x[0], -x[1]))

    # Build the interval tree
    class Node:
        def __init__(self, l, r):
            self.l = l
            self.r = r
            self.children = []
            self.size = 1
            self.res = 1

    root_stack = []

    nodes = []
    for l, r, idx in intervals:
        current_node = Node(l, r)
        nodes.append(current_node)
        while root_stack and not (root_stack[-1].l <= current_node.l and current_node.r <= root_stack[-1].r):
            root_stack.pop()
        if root_stack:
            root_stack[-1].children.append(current_node)
        root_stack.append(current_node)

    # Now, compute the number of permutations recursively
    factorial = [1] * (N+2)
    inv_factorial = [1] * (N+2)
    for i in range(1, N+2):
        factorial[i] = factorial[i-1] * i % MOD
    inv_factorial[N+1] = pow(factorial[N+1], MOD - 2, MOD)
    for i in range(N, -1, -1):
        inv_factorial[i] = inv_factorial[i+1] * (i+1) % MOD

    def dfs(node):
        for child in node.children:
            dfs(child)
            node.size += child.size
        node.res = factorial[node.size -1]
        for child in node.children:
            node.res = node.res * inv_factorial[child.size] % MOD
            node.res = node.res * child.res % MOD

    total_res = 1
    for node in [n for n in nodes if n.l == 1]:
        dfs(node)
        total_res = total_res * node.res % MOD

    print(total_res)



threading.Thread(target=main).start()