class Solution:
    MOD = 10**9 + 7

    def maximumSumSubsequence(self, nums, queries):
        n = len(nums)
        if n == 0:
            return 0
        
        class SegmentTreeNode:
            def __init__(self, l, r):
                self.l = l
                self.r = r
                self.left = None
                self.right = None
                self.t = 0  # max_if_first_taken
                self.n = 0  # max_if_first_not_taken
                self.lt = 0  # max_if_last_taken
                self.ln = 0  # max_if_last_not_taken

            def merge(self, left, right):
                self.t = max(left.t + right.ln, left.n + right.lt)
                self.n = max(left.t + right.ln, left.n + max(right.lt, right.ln))
                self.lt = right.lt + left.ln
                self.ln = max(right.lt, right.ln) + left.ln

        class SegmentTree:
            def __init__(self, data):
                self.n = len(data)
                self.root = self.build(0, self.n - 1, data)

            def build(self, l, r, data):
                node = SegmentTreeNode(l, r)
                if l == r:
                    val = data[l]
                    node.t = val
                    node.n = 0
                    node.lt = val
                    node.ln = 0
                    return node
                mid = (l + r) // 2
                node.left = self.build(l, mid, data)
                node.right = self.build(mid + 1, r, data)
                node.merge(node.left, node.right)
                return node

            def update(self, node, pos, new_val):
                if node.l == node.r == pos:
                    node.t = new_val
                    node.n = 0
                    node.lt = new_val
                    node.ln = 0
                    return
                mid = (node.l + node.r) // 2
                if pos <= mid:
                    self.update(node.left, pos, new_val)
                else:
                    self.update(node.right, pos, new_val)
                node.merge(node.left, node.right)

            def get_max(self):
                return max(self.root.t, self.root.n)

        st = SegmentTree(nums)
        total = 0
        for pos, x in queries:
            nums[pos] = x
            st.update(st.root, pos, x)
            current_max = st.root.t if st.root.t > st.root.n else st.root.n
            total = (total + current_max) % self.MOD
        return total % self.MOD