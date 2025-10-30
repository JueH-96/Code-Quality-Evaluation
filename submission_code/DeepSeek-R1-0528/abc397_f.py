import sys
sys.setrecursionlimit(1500000)

NEG_INF = -10**18

class SegmentTree:
    def __init__(self, data):
        self.n = len(data)
        self.size = 1
        while self.size < self.n:
            self.size *= 2
        self.tree = [0] * (2 * self.size)
        self.lazy = [0] * (2 * self.size)
        for i in range(self.n):
            self.tree[self.size + i] = data[i]
        for i in range(self.n, self.size):
            self.tree[self.size + i] = NEG_INF
        for i in range(self.size-1, 0, -1):
            self.tree[i] = max(self.tree[2*i], self.tree[2*i+1])
    
    def push(self, i, start, end):
        if self.lazy[i] != 0:
            self.tree[i] += self.lazy[i]
            if i < self.size:
                self.lazy[2*i] += self.lazy[i]
                self.lazy[2*i+1] += self.lazy[i]
            self.lazy[i] = 0

    def update_range(self, l, r, delta, i=1, start=0, end=None):
        if end is None:
            end = self.size - 1
        self.push(i, start, end)
        if r < start or end < l:
            return
        if l <= start and end <= r:
            self.lazy[i] += delta
            self.push(i, start, end)
            return
        mid = (start + end) // 2
        self.update_range(l, r, delta, 2*i, start, mid)
        self.update_range(l, r, delta, 2*i+1, mid+1, end)
        self.tree[i] = max(self.tree[2*i], self.tree[2*i+1])
        
    def query_range(self, l, r, i=1, start=0, end=None):
        if end is None:
            end = self.size - 1
        self.push(i, start, end)
        if r < start or end < l:
            return NEG_INF
        if l <= start and end <= r:
            return self.tree[i]
        mid = (start + end) // 2
        left_val = self.query_range(l, r, 2*i, start, mid)
        right_val = self.query_range(l, r, 2*i+1, mid+1, end)
        return max(left_val, right_val)

def main():
    data = sys.stdin.read().split()
    if not data:
        return
    n = int(data[0])
    A = list(map(int, data[1:1+n]))
    
    L = [0] * n
    freq = [0] * (n+1)
    cur = 0
    for i in range(n):
        x = A[i]
        freq[x] += 1
        if freq[x] == 1:
            cur += 1
        L[i] = cur

    R = [0] * (n+1)
    freq = [0] * (n+1)
    cur = 0
    for i in range(n-1, -1, -1):
        x = A[i]
        freq[x] += 1
        if freq[x] == 1:
            cur += 1
        R[i] = cur

    next_occurrence = [n] * n
    temp = [n] * (n+1)
    for i in range(n-1, -1, -1):
        next_occurrence[i] = temp[A[i]]
        temp[A[i]] = i
        
    d_arr = [NEG_INF] * n
    freq2 = [0] * (n+1)
    cur_distinct = 0
    for j0 in range(1, n-1):
        x = A[j0]
        freq2[x] += 1
        if freq2[x] == 1:
            cur_distinct += 1
        d_arr[j0] = cur_distinct + R[j0+1]
    
    seg_tree = SegmentTree(d_arr)
    
    ans = NEG_INF
    for i0 in range(0, n-2):
        q = seg_tree.query_range(i0+1, n-2)
        candidate = L[i0] + q
        if candidate > ans:
            ans = candidate
            
        l_update = i0 + 2
        r_update = min(next_occurrence[i0+1] - 1, n-2)
        if l_update <= r_update:
            seg_tree.update_range(l_update, r_update, -1)
            
    print(ans)

if __name__ == "__main__":
    main()