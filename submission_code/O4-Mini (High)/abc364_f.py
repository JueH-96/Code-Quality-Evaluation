import sys
import threading

def main():
    import sys
    sys.setrecursionlimit(1 << 25)
    data = sys.stdin

    # Read input
    line = data.readline().split()
    if not line:
        print(-1)
        return
    N = int(line[0]); Q = int(line[1])
    intervals = []
    for _ in range(Q):
        l, r, c = map(int, data.readline().split())
        intervals.append((l, r, c))

    # Quick connectivity check on interval graph + coverage
    intervals_L = sorted(intervals, key=lambda x: x[0])
    max_r = -1
    ok = True
    for i, (l, r, _) in enumerate(intervals_L):
        if i == 0:
            if l != 1:
                ok = False
                break
            max_r = r
        else:
            if l > max_r:
                ok = False
                break
            if r > max_r:
                max_r = r
    if (not ok) or max_r < N:
        print(-1)
        return

    # We'll process intervals in increasing cost order and maintain
    # a dynamic set of "active components" represented by disjoint
    # coverage intervals in a treap keyed by left endpoint.
    intervals_C = sorted(intervals, key=lambda x: x[2])

    import random
    def rnd():
        # a random priority
        return random.getrandbits(30)

    class Node:
        __slots__ = ("key","L","R","prio","left","right")
        def __init__(self, L, R):
            self.key = L
            self.L = L
            self.R = R
            self.prio = rnd()
            self.left = None
            self.right = None

    # Treap split/merge/remove/get_max
    def split(root, key):
        if root is None:
            return (None, None)
        if root.key < key:
            # root and left subtree go to "left part"
            l2, r2 = split(root.right, key)
            root.right = l2
            return (root, r2)
        else:
            l1, r1 = split(root.left, key)
            root.left = r1
            return (l1, root)

    def merge(a, b):
        if a is None or b is None:
            return a or b
        if a.prio < b.prio:
            a.right = merge(a.right, b)
            return a
        else:
            b.left = merge(a, b.left)
            return b

    def get_max(node):
        # returns node of maximum key in subtree
        if node is None:
            return None
        while node.right:
            node = node.right
        return node

    def remove_node(root, key):
        # remove exactly one node with root.key == key
        # returns (new_root, removed_subtree)
        # split out [<key], [key..)
        left, midr = split(root, key)
        # split out [key], (>key)
        mid, right = split(midr, key+1)
        # mid is subtree of that single node
        new_root = merge(left, right)
        return new_root, mid

    root = None
    total_cost = 0

    # Iterate intervals in ascending cost
    for (l, r, c) in intervals_C:
        new_L = l
        new_R = r
        length = r - l + 1

        # 1) split out comps with key < l  => left, and >= l => right
        left, right = split(root, l)

        # 2) check if predecessor in 'left' overlaps [l,r]
        covered = 0
        K = 0
        pred = get_max(left)
        if pred is not None and pred.R >= l:
            # remove that pred node
            left, one = remove_node(left, pred.key)
            # 'one' is a subtree containing exactly the pred node
            nd = one
            ovl = nd.L; ovr = nd.R
            covered += min(ovr, r) - max(ovl, l) + 1
            if ovl < new_L: new_L = ovl
            if ovr > new_R: new_R = ovr
            K += 1

        # 3) split 'right' into mid (keys < r+1) and right2 (keys >= r+1)
        mid, right2 = split(right, r+1)

        # 4) traverse all nodes in mid, accumulate overlap and stats
        if mid is not None:
            stack = [mid]
            while stack:
                nd = stack.pop()
                if nd is None:
                    continue
                ovl = nd.L; ovr = nd.R
                covered += min(ovr, r) - max(ovl, l) + 1
                if ovl < new_L: new_L = ovl
                if ovr > new_R: new_R = ovr
                K += 1
                if nd.left: stack.append(nd.left)
                if nd.right: stack.append(nd.right)

        # 5) cost contribution: one union per overlapped comp and per uncovered point
        total_cost += c * (K + (length - covered))

        # 6) merge back: we discard the old overlapped nodes
        #    and insert the new merged interval [new_L,new_R]
        new_node = Node(new_L, new_R)
        left = merge(left, new_node)
        root = merge(left, right2)

    # At this point we know the graph was connected (checked above)
    # so we print the accumulated MST cost
    print(total_cost)

if __name__ == "__main__":
    main()