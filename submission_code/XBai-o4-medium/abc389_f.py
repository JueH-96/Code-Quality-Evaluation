import sys

def main():
    input = sys.stdin.read().split()
    ptr = 0
    N = int(input[ptr])
    ptr += 1

    contests = []
    for _ in range(N):
        L = int(input[ptr])
        R = int(input[ptr + 1])
        contests.append((L, R))
        ptr += 2

    Q = int(input[ptr])
    ptr += 1

    queries = []
    for _ in range(Q):
        queries.append(int(input[ptr]))
        ptr += 1

    U = 700000  # Maximum possible value based on constraints

    class SegmentTree:
        def __init__(self, size_U):
            self.N = 1
            while self.N < size_U:
                self.N <<= 1
            self.size = self.N
            self.tree = [0] * (2 * self.N)
            self.lazy = [0] * (2 * self.N)

        def push(self, node, node_left, node_right):
            if self.lazy[node] != 0 and node < self.N:  # not a leaf
                mid = (node_left + node_right) // 2
                left_node = 2 * node
                right_node = 2 * node + 1
                # propagate to left child
                self.tree[left_node] += self.lazy[node]
                self.lazy[left_node] += self.lazy[node]
                # propagate to right child
                self.tree[right_node] += self.lazy[node]
                self.lazy[right_node] += self.lazy[node]
                # clear the lazy value at the current node
                self.lazy[node] = 0

        def range_add(self, a, b, val, node=1, node_left=1, node_right=None):
            if node_right is None:
                node_right = self.N
            if a > node_right or b < node_left:
                return
            if a <= node_left and node_right <= b:
                self.tree[node] += val
                self.lazy[node] += val
                return
            self.push(node, node_left, node_right)
            mid = (node_left + node_right) // 2
            self.range_add(a, b, val, 2 * node, node_left, mid)
            self.range_add(a, b, val, 2 * node + 1, mid + 1, node_right)

        def point_get(self, pos, node=1, node_left=1, node_right=None):
            if node_right is None:
                node_right = self.N
            if node_left == node_right:
                return self.tree[node]
            self.push(node, node_left, node_right)
            mid = (node_left + node_right) // 2
            if pos <= mid:
                return self.point_get(pos, 2 * node, node_left, mid)
            else:
                return self.point_get(pos, 2 * node + 1, mid + 1, node_right)

    st = SegmentTree(U)

    for L, R in contests:
        # Find y_low
        low = 1
        high = U
        y_low = U + 1
        while low <= high:
            mid = (low + high) // 2
            a_mid = st.point_get(mid)
            f_mid = mid + a_mid
            if f_mid >= L:
                y_low = mid
                high = mid - 1
            else:
                low = mid + 1

        if y_low > U:
            # No valid y_low
            continue

        # Find y_high
        low = 1
        high = U
        y_high = 0
        while low <= high:
            mid = (low + high) // 2
            a_mid = st.point_get(mid)
            f_mid = mid + a_mid
            if f_mid <= R:
                y_high = mid
                low = mid + 1
            else:
                high = mid - 1

        if y_high < 1:
            continue

        if y_low <= y_high:
            st.range_add(y_low, y_high, 1)

    # Process queries
    for X in queries:
        a_X = st.point_get(X)
        print(X + a_X)

if __name__ == "__main__":
    main()