import random
from typing import List

class Solution:
    def numberOfAlternatingGroups(self, colors: List[int], queries: List[List[int]]) -> List[int]:
        n = len(colors)

        # ---------- Fenwick tree ----------
        class Fenwick:
            __slots__ = ('size', 'bit')
            def __init__(self, n: int):
                self.size = n + 2        # 1-based
                self.bit = [0]*(self.size)

            def add(self, idx: int, delta: int):
                while idx < self.size:
                    self.bit[idx] += delta
                    idx += idx & -idx

            def sum(self, idx: int) -> int:          # prefix sum 1..idx
                s = 0
                while idx:
                    s += self.bit[idx]
                    idx -= idx & -idx
                return s
        # ---------- Treap (for zeros positions) ----------
        class Node:
            __slots__ = ('key', 'prio', 'left', 'right')
            def __init__(self, key: int):
                self.key   = key
                self.prio  = random.randint(1, 1 << 30)
                self.left  = None
                self.right = None

        def rotate_left(p: Node) -> Node:
            q = p.right
            p.right = q.left
            q.left  = p
            return q

        def rotate_right(p: Node) -> Node:
            q = p.left
            p.left  = q.right
            q.right = p
            return q

        def treap_insert(root: Node, key: int) -> Node:
            if root is None:
                return Node(key)
            if key < root.key:
                root.left = treap_insert(root.left, key)
                if root.left.prio < root.prio:
                    root = rotate_right(root)
            elif key > root.key:
                root.right = treap_insert(root.right, key)
                if root.right.prio < root.prio:
                    root = rotate_left(root)
            # duplicate key never happens for our use-case
            return root

        def treap_delete(root: Node, key: int) -> Node:
            if root is None:
                return None
            if key < root.key:
                root.left = treap_delete(root.left, key)
            elif key > root.key:
                root.right = treap_delete(root.right, key)
            else:                                       # found node
                if root.left is None:
                    return root.right
                if root.right is None:
                    return root.left
                if root.left.prio < root.right.prio:
                    root = rotate_right(root)
                    root.right = treap_delete(root.right, key)
                else:
                    root = rotate_left(root)
                    root.left = treap_delete(root.left, key)
            return root

        def treap_min(root: Node) -> int:
            while root.left:
                root = root.left
            return root.key

        def treap_max(root: Node) -> int:
            while root.right:
                root = root.right
            return root.key

        def treap_pred(root: Node, key: int) -> int:
            """largest key strictly smaller than key, or wrap-around (max key)"""
            res = None
            cur = root
            while cur:
                if key > cur.key:
                    res = cur.key
                    cur = cur.right
                else:
                    cur = cur.left
            if res is None:
                res = treap_max(root)
            return res

        def treap_succ(root: Node, key: int) -> int:
            """smallest key strictly larger than key, or wrap-around (min key)"""
            res = None
            cur = root
            while cur:
                if key < cur.key:
                    res = cur.key
                    cur = cur.left
                else:
                    cur = cur.right
            if res is None:
                res = treap_min(root)
            return res
        # ---------- helpers ----------
        def dist(a: int, b: int) -> int:
            """ones between zeros b (left) and a (right) clockwise"""
            d = a - b - 1
            if d < 0:
                d += n
            return d

        # ---------- build initial diff, zero set, runs ----------
        diff = [1 if colors[i] != colors[(i+1)%n] else 0 for i in range(n)]
        root = None                       # treap root
        zero_cnt = 0
        for i, v in enumerate(diff):
            if v == 0:
                root = treap_insert(root, i)
                zero_cnt += 1

        runs = []
        i = 0
        while i < n:
            if diff[i] == 1:
                length = 0
                while i < n and diff[i] == 1:
                    length += 1
                    i += 1
                runs.append(length)
            else:
                i += 1
        # merge first & last if they touch circularly
        if diff[0] == 1 and diff[-1] == 1 and len(runs) > 1:
            runs[0] += runs[-1]
            runs.pop()

        fenw_cnt = Fenwick(n)       # frequency of run length
        fenw_len = Fenwick(n)       # sum of lengths
        total_cnt, total_len = 0, 0
        def add_run(L: int):
            nonlocal total_cnt, total_len
            if L == 0:
                return
            fenw_cnt.add(L, 1)
            fenw_len.add(L, L)
            total_cnt += 1
            total_len += L

        def remove_run(L: int):
            nonlocal total_cnt, total_len
            if L == 0:
                return
            fenw_cnt.add(L, -1)
            fenw_len.add(L, -L)
            total_cnt -= 1
            total_len -= L

        for L in runs:
            add_run(L)

        # ---------- diff update routine ----------
        def update_diff(pos: int, new_val: int):
            nonlocal root, zero_cnt
            if diff[pos] == new_val:
                return
            if new_val == 1:                       # 0 -> 1     (remove zero at pos)
                if zero_cnt == 1:                  # only this zero existed
                    remove_run(n-1)                # previous single run
                    add_run(n)                     # whole circle becomes ones
                    root = treap_delete(root, pos)
                    zero_cnt = 0
                else:
                    prev_z = treap_pred(root, pos)
                    next_z = treap_succ(root, pos)
                    L1 = dist(pos, prev_z)
                    L2 = dist(next_z, pos)
                    if L1:
                        remove_run(L1)
                    if L2:
                        remove_run(L2)
                    add_run(L1 + L2 + 1)           # merged run
                    root = treap_delete(root, pos)
                    zero_cnt -= 1
            else:                                  # 1 -> 0     (insert zero at pos)
                if zero_cnt == 0:                  # all ones currently
                    remove_run(n)                  # remove big run
                    add_run(n-1)                   # split into run N-1
                    root = treap_insert(root, pos)
                    zero_cnt = 1
                else:
                    prev_z = treap_pred(root, pos)
                    next_z = treap_succ(root, pos)
                    L_before = dist(next_z, prev_z)
                    remove_run(L_before)
                    L_left  = dist(pos, prev_z)
                    L_right = dist(next_z, pos)
                    if L_left:
                        add_run(L_left)
                    if L_right:
                        add_run(L_right)
                    root = treap_insert(root, pos)
                    zero_cnt += 1
            diff[pos] = new_val

        # ---------- answering queries ----------
        res_out = []
        for q in queries:
            if q[0] == 1:                  # count query
                s = q[1]
                k = s - 1                  # needed length in diff
                if total_cnt == 0:
                    res_out.append(0)
                    continue
                pref_len = fenw_len.sum(k-1)
                pref_cnt = fenw_cnt.sum(k-1)
                s1 = total_len - pref_len
                s2 = total_cnt - pref_cnt
                res_out.append(s1 - (k-1)*s2)
            else:                          # update query
                idx, new_color = q[1], q[2]
                if colors[idx] == new_color:
                    continue
                colors[idx] = new_color
                # affected diff positions
                p1 = (idx - 1) % n
                p2 = idx
                nv1 = 1 if colors[p1] != colors[(p1+1)%n] else 0
                nv2 = 1 if colors[p2] != colors[(p2+1)%n] else 0
                update_diff(p1, nv1)
                if p2 != p1:               # when n == 1 (not possible here) they could coincide
                    update_diff(p2, nv2)

        return res_out