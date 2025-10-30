class Solution:
    def findMaximumLength(self, nums: List[int]) -> int:
        n = len(nums)
        s = [0] * (n + 1)
        for i in range(1, n + 1):
            s[i] = s[i-1] + nums[i-1]
        max_val = 2 * s[n]
        
        def better_state(a, b):
            if a[0] > b[0]:
                return a
            elif a[0] < b[0]:
                return b
            else:
                if a[1] > b[1]:
                    return a
                else:
                    return b
        
        class Node:
            __slots__ = ('l', 'r', 'left', 'right', 'best')
            def __init__(self, l, r):
                self.l = l
                self.r = r
                self.left = None
                self.right = None
                self.best = (0, -1)
        
        def update(node, pos, state):
            if node.l == node.r:
                node.best = better_state(node.best, state)
                return
            mid = (node.l + node.r) // 2
            if pos <= mid:
                if node.left is None:
                    node.left = Node(node.l, mid)
                update(node.left, pos, state)
            else:
                if node.right is None:
                    node.right = Node(mid + 1, node.r)
                update(node.right, pos, state)
            left_best = node.left.best if node.left else (0, -1)
            right_best = node.right.best if node.right else (0, -1)
            node.best = better_state(left_best, right_best)
        
        def query(node, lq, rq):
            if lq <= node.l and node.r <= rq:
                return node.best
            mid = (node.l + node.r) // 2
            res = (0, -1)
            if lq <= mid and node.left is not None:
                left_res = query(node.left, lq, rq)
                res = better_state(res, left_res)
            if rq > mid and node.right is not None:
                right_res = query(node.right, lq, rq)
                res = better_state(res, right_res)
            return res
        
        root = Node(0, max_val)
        update(root, 0, (0, 0))
        
        ans = 0
        for i in range(n):
            s_i = s[i+1]
            state_j = query(root, 0, s_i)
            dp_i = state_j[0] + 1
            g_i = 2 * s_i - state_j[1]
            update(root, g_i, (dp_i, s_i))
            ans = dp_i
        
        return ans