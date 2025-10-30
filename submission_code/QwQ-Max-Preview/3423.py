class Solution:
    def maximumSumSubsequence(self, nums, queries):
        MOD = 10**9 + 7
        
        class SegmentTree:
            def __init__(self, data):
                self.n = len(data)
                self.tree = []
                self.build(0, self.n - 1, 0, data)
            
            def build(self, l, r, node, data):
                if l > r:
                    return
                while node >= len(self.tree):
                    self.tree.append((0, 0))
                if l == r:
                    self.tree[node] = (data[l], 0)
                    return
                mid = (l + r) // 2
                self.build(l, mid, 2 * node + 1, data)
                self.build(mid + 1, r, 2 * node + 2, data)
                left = self.tree[2 * node + 1]
                right = self.tree[2 * node + 2]
                take = max(left[0] + right[1], left[1] + right[0])
                not_take = max(left[0] + right[1], left[1] + right[1])
                self.tree[node] = (take, not_take)
            
            def update(self, pos, new_val, l, r, node):
                if l == r:
                    self.tree[node] = (new_val, 0)
                    return
                mid = (l + r) // 2
                if pos <= mid:
                    self.update(pos, new_val, l, mid, 2 * node + 1)
                else:
                    self.update(pos, new_val, mid + 1, r, 2 * node + 2)
                left_node = 2 * node + 1
                right_node = 2 * node + 2
                left = self.tree[left_node] if left_node < len(self.tree) else (-float('inf'), -float('inf'))
                right = self.tree[right_node] if right_node < len(self.tree) else (-float('inf'), -float('inf'))
                take = max(left[0] + right[1], left[1] + right[0])
                not_take = max(left[0] + right[1], left[1] + right[1])
                self.tree[node] = (take, not_take)
            
            def get_max(self):
                return max(self.tree[0][0], self.tree[0][1], 0)
        
        if not nums:
            return 0
        st = SegmentTree(nums)
        result = 0
        for pos, x in queries:
            st.update(pos, x, 0, st.n - 1, 0)
            current_max = st.get_max()
            result = (result + current_max) % MOD
        return result % MOD