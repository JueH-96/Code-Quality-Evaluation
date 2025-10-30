INF_NEG = -10**18
mod = 10**9 + 7

class Solution:
    def maximumSumSubsequence(self, nums: list, queries: list) -> int:
        n = len(nums)
        size = 4 * n
        tree = [(0, 0, 0, 0)] * size
        
        def merge_node(l, r):
            l00, l01, l10, l11 = l
            r00, r01, r10, r11 = r
            res = [INF_NEG] * 4
            l_vals = [l00, l01, l10, l11]
            r_vals = [r00, r01, r10, r11]
            for a in range(4):
                for b in range(4):
                    if (a in (1, 3) and b in (2, 3)):
                        continue
                    val = l_vals[a] + r_vals[b]
                    left_first = a // 2
                    right_last = b % 2
                    idx = left_first * 2 + right_last
                    if val > res[idx]:
                        res[idx] = val
            return (res[0], res[1], res[2], res[3])
        
        def build_tree(id, l, r):
            if l == r:
                tree[id] = (0, INF_NEG, INF_NEG, nums[l])
            else:
                mid = (l + r) // 2
                build_tree(2*id + 1, l, mid)
                build_tree(2*id + 2, mid+1, r)
                tree[id] = merge_node(tree[2*id+1], tree[2*id+2])
        
        def update_tree(id, l, r, pos, val):
            if l == r:
                tree[id] = (0, INF_NEG, INF_NEG, val)
            else:
                mid = (l + r) // 2
                if pos <= mid:
                    update_tree(2*id+1, l, mid, pos, val)
                else:
                    update_tree(2*id+2, mid+1, r, pos, val)
                tree[id] = merge_node(tree[2*id+1], tree[2*id+2])
        
        build_tree(0, 0, n-1)
        res = 0
        for pos, x in queries:
            update_tree(0, 0, n-1, pos, x)
            current_ans = max(tree[0])
            res = (res + current_ans) % mod
        return res