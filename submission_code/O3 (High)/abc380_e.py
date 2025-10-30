import sys, random

# ----------  Treap implementation on interval starts  ---------- #
class Node:
    __slots__ = ('l', 'r', 'col', 'prio', 'left', 'right')
    def __init__(self, l, r, col):
        self.l = l          # interval left edge  (also the key)
        self.r = r          # interval right edge
        self.col = col      # colour of the interval
        self.prio = random.randint(1, 1 << 30)
        self.left = None
        self.right = None


def rotate_right(p):
    q = p.left
    p.left = q.right
    q.right = p
    return q


def rotate_left(p):
    q = p.right
    p.right = q.left
    q.left = p
    return q


def insert(root, node):
    if root is None:
        return node
    if node.l < root.l:
        root.left = insert(root.left, node)
        if root.left.prio < root.prio:
            root = rotate_right(root)
    else:
        root.right = insert(root.right, node)
        if root.right.prio < root.prio:
            root = rotate_left(root)
    return root


def _delete_root(root):
    if root.left is None:
        return root.right
    if root.right is None:
        return root.left
    if root.left.prio < root.right.prio:
        root = rotate_right(root)
        root.right = _delete_root(root.right)
    else:
        root = rotate_left(root)
        root.left = _delete_root(root.left)
    return root


def erase(root, key):
    if root is None:
        return None
    if key < root.l:
        root.left = erase(root.left, key)
    elif key > root.l:
        root.right = erase(root.right, key)
    else:
        root = _delete_root(root)
    return root


def find_le(root, x):          # greatest l  ≤  x
    res = None
    while root:
        if x < root.l:
            root = root.left
        else:
            res = root
            root = root.right
    return res


def predecessor(root, key):    # greatest l  <  key
    res = None
    while root:
        if key <= root.l:
            root = root.left
        else:
            res = root
            root = root.right
    return res


def successor(root, key):      # smallest l  >  key
    res = None
    while root:
        if root.l <= key:
            root = root.right
        else:
            res = root
            root = root.left
    return res


# ----------  main logic  ---------- #
def main() -> None:
    sys.setrecursionlimit(1_000_000)
    random.seed(0)

    it = iter(sys.stdin.buffer.read().split())
    n = int(next(it))
    q = int(next(it))

    # colour counters
    cnt = [0]*(n+1)
    for i in range(1, n+1):
        cnt[i] = 1

    # build initial treap: each cell its own interval
    root = None
    for i in range(1, n+1):
        root = insert(root, Node(i, i, i))

    out_lines = []

    for _ in range(q):
        t = int(next(it))
        if t == 1:
            x = int(next(it))
            c = int(next(it))

            node = find_le(root, x)
            # x must be inside that interval
            if node.col == c:
                continue

            seg_len = node.r - node.l + 1
            cnt[node.col] -= seg_len
            cnt[c] += seg_len
            node.col = c

            # merge with left neighbour if same colour
            pred = predecessor(root, node.l)
            if pred and pred.r == node.l - 1 and pred.col == node.col:
                pred.r = node.r
                root = erase(root, node.l)
                node = pred

            # merge with right neighbour if same colour
            succ = successor(root, node.l)
            if succ and node.r + 1 == succ.l and succ.col == node.col:
                node.r = succ.r
                root = erase(root, succ.l)

        else:        # type-2
            c = int(next(it))
            out_lines.append(str(cnt[c]))

    sys.stdout.write('
'.join(out_lines))


if __name__ == "__main__":
    main()