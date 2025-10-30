class SegmentTreeNode:
    def __init__(self, l, r):
        self.l = l
        self.r = r
        self.left = None
        self.right = None
        self.peak_count = 0
        self.left_val = 0
        self.right_val = 0

class SegmentTree:
    def __init__(self, data):
        self.n = len(data)
        self.root = self.build(0, self.n - 1, data)
    
    def build(self, l, r, data):
        node = SegmentTreeNode(l, r)
        if l == r:
            node.peak_count = 0
            node.left_val = data[l]
            node.right_val = data[l]
        else:
            mid = (l + r) // 2
            node.left = self.build(l, mid, data)
            node.right = self.build(mid + 1, r, data)
            self._combine(node)
        return node
    
    def _combine(self, node):
        left_res = node.left.peak_count
        left_val = node.left.left_val
        right_val = node.left.right_val
        
        right_res = node.right.peak_count
        right_left = node.right.left.left_val
        right_right = node.right.right.right_val
        
        peak_count = left_res + right_res
        if node.right.left.left_val > left_val and node.right.left.left_val > right_val:
            peak_count += 1
        
        node.peak_count = peak_count
        node.left_val = left_val
        node.right_val = right_val
    
    def _update(self, node, idx, new_val):
        if node.l == node.r == idx:
            node.left_val = new_val
            node.right_val = new_val
            return
        mid = (node.l + node.r) // 2
        if idx <= mid:
            self._update(node.left, idx, new_val)
        else:
            self._update(node.right, idx, new_val)
        self._combine(node)
    
    def _query(self, node, l, r):
        if node.r < l or node.l > r:
            return (0, 0, 0)
        if l <= node.l and node.r <= r:
            return (node.peak_count, node.left_val, node.right_val)
        left_res = self._query(node.left, l, r)
        right_res = self._query(node.right, l, r)
        peak_count = left_res[0] + right_res[0]
        if right_res[1] > left_res[2] and right_res[1] > right_res[2]:
            peak_count += 1
        return (peak_count, left_res[1], right_res[2])

def countOfPeaks(nums, queries):
    if not nums:
        return []
    st = SegmentTree(nums)
    res = []
    for q in queries:
        if q[0] == 1:
            idx = q[1]
            new_val = q[2]
            st._update(st.root, idx, new_val)
        else:
            l = q[1]
            r = q[2]
            count = st._query(st.root, l, r)[0]
            res.append(count)
    return res