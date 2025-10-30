import sys

class SegmentTree:
    def __init__(self, data):
        self.n = len(data)
        self.size = 1
        while self.size < self.n:
            self.size *= 2
        self.tree = [-10**18] * (2 * self.size)
        self.lazy = [0] * (2 * self.size)
        for i in range(self.n):
            self.tree[self.size + i] = data[i]
        for i in range(self.n, self.size):
            self.tree[self.size + i] = -10**18
        for i in range(self.size - 1, 0, -1):
            self.tree[i] = max(self.tree[2*i], self.tree[2*i+1])
    
    def push(self, idx):
        if self.lazy[idx] != 0:
            self.tree[2*idx] += self.lazy[idx]
            self.tree[2*idx+1] += self.lazy[idx]
            if 2*idx < self.size:
                self.lazy[2*idx] += self.lazy[idx]
                self.lazy[2*idx+1] += self.lazy[idx]
            self.lazy[idx] = 0

    def update_range(self, l, r, val, idx=1, segL=0, segR=None):
        if segR is None:
            segR = self.size - 1
        if r < segL or l > segR:
            return
        if l <= segL and segR <= r:
            self.tree[idx] += val
            if idx < self.size:
                self.lazy[idx] += val
            return
        self.push(idx)
        mid = (segL + segR) // 2
        self.update_range(l, r, val, 2*idx, segL, mid)
        self.update_range(l, r, val, 2*idx+1, mid+1, segR)
        self.tree[idx] = max(self.tree[2*idx], self.tree[2*idx+1])
    
    def query(self, l, r, idx=1, segL=0, segR=None):
        if segR is None:
            segR = self.size - 1
        if r < segL or l > segR:
            return -10**18
        if l <= segL and segR <= r:
            return self.tree[idx]
        self.push(idx)
        mid = (segL + segR) // 2
        left_res = self.query(l, r, 2*idx, segL, mid)
        right_res = self.query(l, r, 2*idx+1, mid+1, segR)
        return max(left_res, right_res)

def main():
    data = sys.stdin.read().split()
    if not data:
        return
    
    n = int(data[0])
    A = list(map(int, data[1:1+n]))
    
    L = [0] * (n+1)
    freq = [0] * (n+1)
    distinct_count = 0
    for i in range(n):
        if freq[A[i]] == 0:
            distinct_count += 1
        freq[A[i]] += 1
        L[i+1] = distinct_count
        
    R = [0] * (n+1)
    freq = [0] * (n+1)
    distinct_count = 0
    for i in range(n-1, -1, -1):
        if freq[A[i]] == 0:
            distinct_count += 1
        freq[A[i]] += 1
        R[i] = distinct_count
        
    last_occ = [-1] * (n+1)
    prev_arr = [-1] * n
    for i in range(n):
        if last_occ[A[i]] != -1:
            prev_arr[i] = last_occ[A[i]]
        else:
            prev_arr[i] = -1
        last_occ[A[i]] = i
        
    init_data = [0] * n
    for k in range(n):
        init_data[k] = L[k]
        
    seg_tree = SegmentTree(init_data)
    
    ans = 0
    for j in range(n):
        p = prev_arr[j]
        low = p + 1
        high = j
        if low <= high:
            seg_tree.update_range(low, high, 1)
            
        if j >= 1 and j <= n-2:
            max_val = seg_tree.query(1, j)
            candidate = max_val + R[j+1]
            if candidate > ans:
                ans = candidate
                
    print(ans)

if __name__ == "__main__":
    main()