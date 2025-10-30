import sys

class SegmentTree:
    def __init__(self, size):
        self.n = 1
        while self.n < size:
            self.n <<= 1
        self.size = self.n
        self.min_tree = [float('inf')] * (2 * self.n)
        self.lazy = [float('inf')] * (2 * self.n)

    def push(self, node, node_l, node_r):
        if self.lazy[node] != float('inf'):
            mid = (node_l + node_r) // 2
            left = 2 * node
            right_node = 2 * node + 1
            # Update left child
            self.min_tree[left] = min(self.min_tree[left], self.lazy[node])
            self.lazy[left] = min(self.lazy[left], self.lazy[node])
            # Update right child
            self.min_tree[right_node] = min(self.min_tree[right_node], self.lazy[node])
            self.lazy[right_node] = min(self.lazy[right_node], self.lazy[node])
            # Clear the lazy value
            self.lazy[node] = float('inf')

    def update_range(self, a, b, val, node=1, node_l=1, node_r=None):
        if node_r is None:
            node_r = self.n
        if a > node_r or b < node_l:
            return
        if a <= node_l and node_r <= b:
            if val < self.min_tree[node]:
                self.min_tree[node] = val
                if node_l != node_r:
                    self.lazy[node] = min(self.lazy[node], val)
            return
        self.push(node, node_l, node_r)
        mid = (node_l + node_r) // 2
        self.update_range(a, b, val, 2*node, node_l, mid)
        self.update_range(a, b, val, 2*node+1, mid+1, node_r)
        self.min_tree[node] = min(self.min_tree[2*node], self.min_tree[2*node+1])

    def query_point(self, idx, node=1, node_l=1, node_r=None):
        if node_r is None:
            node_r = self.n
        if node_l == node_r:
            return self.min_tree[node]
        self.push(node, node_l, node_r)
        mid = (node_l + node_r) // 2
        if idx <= mid:
            return self.query_point(idx, 2*node, node_l, mid)
        else:
            return self.query_point(idx, 2*node+1, mid+1, node_r)

def main():
    input = sys.stdin.read().split()
    ptr = 0
    N = int(input[ptr])
    ptr += 1
    M = int(input[ptr])
    ptr += 1
    A = list(map(int, input[ptr:ptr+N]))
    ptr += N
    B = list(map(int, input[ptr:ptr+M]))
    ptr += M

    max_B = 200000
    st = SegmentTree(max_B)

    for i in range(N):
        current_A = A[i]
        current_person = i + 1
        st.update_range(current_A, max_B, current_person)

    for b in B:
        res = st.query_point(b)
        print(-1 if res == float('inf') else res)

if __name__ == "__main__":
    main()