import sys

class SegmentTree:
    def __init__(self, data):
        self.n = len(data)
        self.size = 1
        while self.size < self.n:
            self.size <<= 1
        self.min_tree = [float('inf')] * (2 * self.size)
        for i in range(self.n):
            self.min_tree[self.size + i] = data[i]
        for i in range(self.size - 1, 0, -1):
            self.min_tree[i] = min(self.min_tree[2*i], self.min_tree[2*i+1])
    
    def update(self, pos, value):
        pos += self.size
        self.min_tree[pos] = value
        pos >>= 1
        while pos >= 1:
            new_val = min(self.min_tree[2*pos], self.min_tree[2*pos+1])
            if self.min_tree[pos] == new_val:
                break
            self.min_tree[pos] = new_val
            pos >>= 1
    
    def query_leftmost(self, T):
        return self._query(1, 0, self.size-1, T)
    
    def _query(self, node, node_l, node_r, T):
        if node_l == node_r:
            return node_l if self.min_tree[node] <= T else -1
        mid = (node_l + node_r) // 2
        if self.min_tree[2*node] <= T:
            left_res = self._query(2*node, node_l, mid, T)
            if left_res != -1:
                return left_res
        if self.min_tree[2*node+1] <= T:
            return self._query(2*node+1, mid+1, node_r, T)
        return -1

def main():
    input = sys.stdin.read().split()
    ptr = 0
    N = int(input[ptr])
    ptr +=1
    M = int(input[ptr])
    ptr +=1
    events = []
    for _ in range(M):
        T = int(input[ptr])
        ptr +=1
        W = int(input[ptr])
        ptr +=1
        S = int(input[ptr])
        ptr +=1
        events.append( (T, W, S) )
    
    available_time = [0] * N
    st = SegmentTree(available_time)
    ans = [0] * N
    
    for t, w, s in events:
        i = st.query_leftmost(t)
        if i != -1:
            ans[i] += w
            st.update(i, t + s)
    
    for a in ans:
        print(a)

if __name__ == "__main__":
    main()