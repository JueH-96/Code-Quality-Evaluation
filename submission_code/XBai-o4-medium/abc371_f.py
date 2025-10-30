class SegmentTreeSum:
    def __init__(self, data):
        self.n = len(data)
        self.size = 1
        while self.size < self.n:
            self.size <<= 1
        self.tree = [0] * (2 * self.size)
        self.lazy = [None] * (2 * self.size)
        # Initialize leaves
        for i in range(self.n):
            self.tree[self.size + i] = data[i]
        # Build the tree
        for i in range(self.size - 1, 0, -1):
            self.tree[i] = self.tree[2 * i] + self.tree[2 * i + 1]

    def _push(self, node, l, r):
        if self.lazy[node] is not None:
            mid = (l + r) // 2
            left_node = 2 * node
            right_node = 2 * node + 1
            # Propagate to left child
            self.tree[left_node] = self.lazy[node] * (mid - l + 1)
            self.lazy[left_node] = self.lazy[node]
            # Propagate to right child
            self.tree[right_node] = self.lazy[node] * (r - mid)
            self.lazy[right_node] = self.lazy[node]
            # Clear the lazy value
            self.lazy[node] = None

    def range_assign(self, a, b, val, node=1, l=0, r=None):
        if r is None:
            r = self.size - 1
        if a > r or b < l:
            return
        if a <= l and r <= b:
            self.tree[node] = val * (r - l + 1)
            self.lazy[node] = val
            return
        self._push(node, l, r)
        mid = (l + r) // 2
        self.range_assign(a, b, val, 2 * node, l, mid)
        self.range_assign(a, b, val, 2 * node + 1, mid + 1, r)
        self.tree[node] = self.tree[2 * node] + self.tree[2 * node + 1]

    def range_sum(self, a, b, node=1, l=0, r=None):
        if r is None:
            r = self.size - 1
        if a > r or b < l:
            return 0
        if a <= l and r <= b:
            return self.tree[node]
        self._push(node, l, r)
        mid = (l + r) // 2
        return self.range_sum(a, b, 2 * node, l, mid) + self.range_sum(a, b, 2 * node + 1, mid + 1, r)


class SegmentTreePoint:
    def __init__(self, data):
        self.n = len(data)
        self.size = 1
        while self.size < self.n:
            self.size <<= 1
        self.tree = [0] * (2 * self.size)
        self.lazy = [None] * (2 * self.size)
        self.original_data = data
        # Initialize leaves
        for i in range(self.n):
            self.tree[self.size + i] = data[i]
        # Build the tree (not needed for point queries)
        for i in range(self.size - 1, 0, -1):
            self.tree[i] = self.tree[2 * i] + self.tree[2 * i + 1]

    def _push(self, node, l, r):
        if self.lazy[node] is not None:
            mid = (l + r) // 2
            left_node = 2 * node
            right_node = 2 * node + 1
            # Propagate to children
            self.lazy[left_node] = self.lazy[node]
            self.lazy[right_node] = self.lazy[node]
            # Clear the lazy value
            self.lazy[node] = None

    def range_assign(self, a, b, val, node=1, l=0, r=None):
        if r is None:
            r = self.size - 1
        if a > r or b < l:
            return
        if a <= l and r <= b:
            self.lazy[node] = val
            return
        self._push(node, l, r)
        mid = (l + r) // 2
        self.range_assign(a, b, val, 2 * node, l, mid)
        self.range_assign(a, b, val, 2 * node + 1, mid + 1, r)

    def point_query(self, idx, node=1, l=0, r=None):
        if r is None:
            r = self.size - 1
        if l == r:
            if self.lazy[node] is not None:
                return (l + 1) + self.lazy[node]
            else:
                return self.original_data[l]
        self._push(node, l, r)
        mid = (l + r) // 2
        if idx <= mid:
            return self.point_query(idx, 2 * node, l, mid)
        else:
            return self.point_query(idx, 2 * node + 1, mid + 1, r)


def main():
    import sys
    input = sys.stdin.read().split()
    ptr = 0
    N = int(input[ptr])
    ptr += 1
    X = list(map(int, input[ptr:ptr + N]))
    ptr += N
    data_st2 = [X[i] - (i + 1) for i in range(N)]
    st_sum = SegmentTreeSum(data_st2)
    st_point = SegmentTreePoint(X)
    Q = int(input[ptr])
    ptr += 1
    total_movement = 0
    for _ in range(Q):
        T = int(input[ptr])
        G = int(input[ptr + 1])
        ptr += 2
        k = T
        C = G - k
        # Query current_pos[k]
        current_pos_k = st_point.point_query(k - 1)
        current_pos_k_minus_k = current_pos_k - k
        sum_left = 0
        a = None
        if current_pos_k_minus_k >= C:
            low = 1
            high = k
            res_a = k
            while low <= high:
                mid = (low + high) // 2
                val = st_point.point_query(mid - 1) - mid
                if val >= C:
                    res_a = mid
                    high = mid - 1
                else:
                    low = mid + 1
            a = res_a
            sum_val = st_sum.range_sum(a - 1, k - 1)
            count = k - a + 1
            sum_left = sum_val - C * count
        else:
            sum_left = 0
        sum_right = 0
        b = None
        if current_pos_k_minus_k <= C:
            low = k
            high = N
            res_b = k
            while low <= high:
                mid = (low + high) // 2
                val = st_point.point_query(mid - 1) - mid
                if val <= C:
                    res_b = mid
                    low = mid + 1
                else:
                    high = mid - 1
            b = res_b
            sum_val = st_sum.range_sum(k - 1, b - 1)
            count = b - k + 1
            sum_right = C * count - sum_val
        else:
            sum_right = 0
        total_movement += sum_left + sum_right
        if a is not None and b is not None:
            st_point.range_assign(a - 1, b - 1, C)
            st_sum.range_assign(a - 1, b - 1, C)
        elif a is not None:
            st_point.range_assign(a - 1, k - 1, C)
            st_sum.range_assign(a - 1, k - 1, C)
        elif b is not None:
            st_point.range_assign(k - 1, b - 1, C)
            st_sum.range_assign(k - 1, b - 1, C)
    print(total_movement)


if __name__ == "__main__":
    main()