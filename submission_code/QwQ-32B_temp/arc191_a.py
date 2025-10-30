import sys

class SegmentTree:
    def __init__(self, data):
        self.n = len(data)
        self.size = 1
        while self.size < self.n:
            self.size <<= 1
        self.min_tree = [float('inf')] * (2 * self.size)
        # Fill the leaves
        for i in range(self.n):
            self.min_tree[self.size + i] = int(data[i])
        # Build the tree
        for i in range(self.size - 1, 0, -1):
            self.min_tree[i] = min(self.min_tree[2 * i], self.min_tree[2 * i + 1])

    def update(self, pos, value):
        pos += self.size
        self.min_tree[pos] = value
        pos >>= 1
        while pos >= 1:
            new_val = min(self.min_tree[2 * pos], self.min_tree[2 * pos + 1])
            if self.min_tree[pos] == new_val:
                break
            self.min_tree[pos] = new_val
            pos >>= 1

    def query_first_less(self, x):
        return self._find_first_less(1, 0, self.size - 1, x)

    def _find_first_less(self, node, l, r, x):
        if l == r:
            if self.min_tree[node] < x:
                return l
            else:
                return -1
        mid = (l + r) // 2
        left_node = 2 * node
        right_node = 2 * node + 1
        left_min = self.min_tree[left_node]
        if left_min < x:
            res = self._find_first_less(left_node, l, mid, x)
            if res != -1:
                return res
        right_min = self.min_tree[right_node]
        if right_min < x:
            res = self._find_first_less(right_node, mid + 1, r, x)
            return res
        return -1

def main():
    input = sys.stdin.read().split()
    N = int(input[0])
    M = int(input[1])
    S = input[2]
    T = input[3]

    data = list(S)
    st = SegmentTree(data)
    s_list = list(data)

    for t_char in T:
        x = int(t_char)
        pos = st.query_first_less(x)
        if pos != -1:
            # Replace the found position
            s_list[pos] = str(x)
            st.update(pos, x)
        else:
            # Replace the last position
            pos = N - 1
            s_list[pos] = str(x)
            st.update(pos, x)

    print(''.join(s_list))

if __name__ == '__main__':
    main()