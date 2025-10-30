import sys
sys.setrecursionlimit(10**6)

class Solution:
    def maximumSumSubsequence(self, nums, queries):
        mod = 10**9 + 7
        n = len(nums)
        if n == 0:
            return 0
        NEG_INF = -10**18
        size = 4 * n
        tree = [(0, 0, 0, 0)] * size
        
        def merge(L, R):
            aL, bL, cL, dL = L
            aR, bR, cR, dR = R
            a = max(aL + aR, aL + cR, bL + aR)
            b = max(aL + bR, aL + dR, bL + bR)
            c = max(cL + aR, cL + cR, dL + aR)
            d = max(cL + bR, cL + dR, dL + bR)
            return (a, b, c, d)
        
        def build_tree(idx, l, r):
            if l == r:
                tree[idx] = (0, NEG_INF, NEG_INF, nums[l])
            else:
                mid = (l + r) // 2
                build_tree(2 * idx + 1, l, mid)
                build_tree(2 * idx + 2, mid + 1, r)
                tree[idx] = merge(tree[2 * idx + 1], tree[2 * idx + 2])
        
        def update_tree(idx, l, r, pos, x):
            if l == r:
                tree[idx] = (0, NEG_INF, NEG_INF, x)
            else:
                mid = (l + r) // 2
                if pos <= mid:
                    update_tree(2 * idx + 1, l, mid, pos, x)
                else:
                    update_tree(2 * idx + 2, mid + 1, r, pos, x)
                tree[idx] = merge(tree[2 * idx + 1], tree[2 * idx + 2])
        
        build_tree(0, 0, n - 1)
        total_ans = 0
        for pos, x in queries:
            update_tree(0, 0, n - 1, pos, x)
            root_node = tree[0]
            current_ans = max(root_node)
            total_ans = (total_ans + current_ans) % mod
        
        return total_ans % mod