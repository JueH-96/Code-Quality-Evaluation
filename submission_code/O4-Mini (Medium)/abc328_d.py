import sys
import threading
def main():
    import sys
    import heapq

    sys.setrecursionlimit(1 << 25)
    S = sys.stdin.readline().strip()
    n = len(S)

    # Node for a doubly-linked list
    class Node:
        __slots__ = ('char','prev','next','index','dead')
        def __init__(self, ch, prev, nxt, idx):
            self.char = ch
            self.prev = prev
            self.next = nxt
            self.index = idx
            self.dead = False

    # Build the list with a dummy head
    dummy = Node('', None, None, -1)
    prev = dummy
    nodes = []
    for i, ch in enumerate(S):
        node = Node(ch, prev, None, i)
        prev.next = node
        prev = node
        nodes.append(node)
    # tail = prev  # not needed explicitly

    # Min-heap of (start_index, node) for each "ABC" occurrence
    heap = []
    def try_push(node):
        # If node, node.next, node.next.next form "ABC" and are alive & consecutive, push
        if node is None or node.dead:
            return
        y = node.next
        if y is None or y.dead:
            return
        z = y.next
        if z is None or z.dead:
            return
        # Must be direct links
        if node.next is not y or y.next is not z:
            return
        if node.char == 'A' and y.char == 'B' and z.char == 'C':
            heapq.heappush(heap, (node.index, node))

    # Initialize with all initial ABCs
    cur = dummy.next
    while cur and cur.next and cur.next.next:
        if cur.char == 'A' and cur.next.char == 'B' and cur.next.next.char == 'C':
            heapq.heappush(heap, (cur.index, cur))
        cur = cur.next

    # Process removals
    while heap:
        idx, x = heapq.heappop(heap)
        # Validate that x, x.next, x.next.next are still alive and "ABC"
        if x.dead:
            continue
        y = x.next
        if y is None or y.dead:
            continue
        z = y.next
        if z is None or z.dead:
            continue
        if x.next is not y or y.next is not z:
            continue
        if not (x.char == 'A' and y.char == 'B' and z.char == 'C'):
            continue
        # Remove x, y, z
        pre = x.prev
        post = z.next
        x.dead = True
        y.dead = True
        z.dead = True
        # Relink
        pre.next = post
        if post:
            post.prev = pre
        # New candidate starts can be at pre.prev, pre, post
        try_push(pre.prev)
        try_push(pre)
        try_push(post)

    # Collect result by traversing from dummy
    res = []
    cur = dummy.next
    while cur:
        # All dead nodes have been unlinked, no need to check dead flag
        res.append(cur.char)
        cur = cur.next

    sys.stdout.write(''.join(res))

if __name__ == "__main__":
    main()