class SegmentTree:
    def __init__(self, data):
        self.n = len(data)
        self.incl = [0] * (4 * self.n)
        self.excl = [0] * (4 * self.n)
        self.build(0, 0, self.n - 1, data)

    def build(self, node, l, r, data):
        if l == r:
            val = data[l]
            self.incl[node] = max(0, val)
            self.excl[node] = 0
        else:
            mid = (l + r) // 2
            left_node = 2 * node + 1
            right_node = 2 * node + 2
            self.build(left_node, l, mid, data)
            self.build(right_node, mid + 1, r, data)
            self.incl[node] = self.incl[left_node] + self.excl[right_node]
            self.excl[node] = max(self.incl[left_node], self.excl[left_node]) + max(self.incl[right_node], self.excl[right_node])

    def update_val(self, node, l, r, idx, val):
        if l == r:
            self.incl[node] = max(0, val)
            self.excl[node] = 0
        else:
            mid = (l + r) // 2
            if idx <= mid:
                self.update_val(2 * node + 1, l, mid, idx, val)
            else:
                self.update_val(2 * node + 2, mid + 1, r, idx, val)
            left_node = 2 * node + 1
            right_node = 2 * node + 2
            self.incl[node] = self.incl[left_node] + self.excl[right_node]
            self.excl[node] = max(self.incl[left_node], self.excl[left_node]) + max(self.incl[right_node], self.excl[right_node])

    def get_max(self):
        return max(self.incl[0], self.excl[0])

def maximumSumSubsequence(nums: List[int], queries: List[List[int]]) -> int:
    mod = 10**9 + 7
    st = SegmentTree(nums)
    res = 0
    for pos, x in queries:
        arr = nums  # Since we're using the same list, but in the SegmentTree, it's passed as data
        arr[pos] = x
        st.update_val(0, 0, len(arr) - 1, pos, x)
        current_max = st.get_max()
        res = (res + current_max) % mod
    return res