class Solution:
    def minAbsoluteDifference(self, nums: List[int], x: int) -> int:
        import random

        class TreapNode:
            __slots__ = 'key', 'priority', 'left', 'right'

            def __init__(self, key):
                self.key = key
                self.priority = random.randint(1, 1 << 30)
                self.left = None
                self.right = None

        def rotate_left(p):
            r = p.right
            p.right = r.left
            r.left = p
            return r

        def rotate_right(p):
            l = p.left
            p.left = l.right
            l.right = p
            return l

        def treap_insert(p, key):
            if p is None:
                return TreapNode(key)
            if key < p.key:
                p.left = treap_insert(p.left, key)
                if p.left.priority > p.priority:
                    p = rotate_right(p)
            else:
                p.right = treap_insert(p.right, key)
                if p.right.priority > p.priority:
                    p = rotate_left(p)
            return p

        def treap_find_predecessor(p, key):
            pred = None
            while p:
                if key > p.key:
                    pred = p.key
                    p = p.right
                else:
                    p = p.left
            return pred

        def treap_find_successor(p, key):
            succ = None
            while p:
                if key < p.key:
                    succ = p.key
                    p = p.left
                else:
                    p = p.right
            return succ

        n = len(nums)
        ans = float('inf')
        root = None

        for i in range(x, n):
            key = nums[i - x]
            root = treap_insert(root, key)

            pred = treap_find_predecessor(root, nums[i])
            if pred is not None:
                ans = min(ans, abs(nums[i] - pred))
            succ = treap_find_successor(root, nums[i])
            if succ is not None:
                ans = min(ans, abs(nums[i] - succ))

        return ans