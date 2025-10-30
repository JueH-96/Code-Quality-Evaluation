import sys

INF = 10**18

class SegmentTree:
    def __init__(self, n, arr):
        self.n = n
        self.arr = arr
        self.size = 4 * n
        self.min_val = [0] * self.size
        self.max_val = [0] * self.size
        self.sum_val = [0] * self.size
        self.lazy_ceiling = [INF] * self.size
        self.actual_size = [0] * self.size
        self.build(0, 0, n-1)
    
    def build(self, node, l, r):
        if l == r:
            self.min_val[node] = self.arr[l]
            self.max_val[node] = self.arr[l]
            self.sum_val[node] = self.arr[l]
            self.actual_size[node] = 1
            self.lazy_ceiling[node] = INF
            return
        mid = (l + r) // 2
        left_node = 2*node+1
        right_node = 2*node+2
        self.build(left_node, l, mid)
        self.build(right_node, mid+1, r)
        self.min_val[node] = min(self.min_val[left_node], self.min_val[right_node])
        self.max_val[node] = max(self.max_val[left_node], self.max_val[right_node])
        self.sum_val[node] = self.sum_val[left_node] + self.sum_val[right_node]
        self.actual_size[node] = self.actual_size[left_node] + self.actual_size[right_node]
        self.lazy_ceiling[node] = INF
        
    def push(self, node, l, r):
        if self.lazy_ceiling[node] != INF:
            ceiling_val = self.lazy_ceiling[node]
            if l == r:
                if ceiling_val < self.min_val[node]:
                    self.min_val[node] = ceiling_val
                    self.max_val[node] = ceiling_val
                    self.sum_val[node] = ceiling_val
                self.lazy_ceiling[node] = INF
                return
            left_node = 2*node+1
            right_node = 2*node+2
            mid = (l + r) // 2
            if ceiling_val < self.min_val[node]:
                self.min_val[node] = ceiling_val
                self.max_val[node] = ceiling_val
                self.sum_val[node] = ceiling_val * self.actual_size[node]
                self.lazy_ceiling[left_node] = min(self.lazy_ceiling[left_node], ceiling_val)
                self.lazy_ceiling[right_node] = min(self.lazy_ceiling[right_node], ceiling_val)
            elif ceiling_val < self.max_val[node]:
                self.apply(left_node, l, mid, ceiling_val)
                self.apply(right_node, mid+1, r, ceiling_val)
                self.min_val[node] = min(self.min_val[left_node], self.min_val[right_node])
                self.max_val[node] = max(self.max_val[left_node], self.max_val[right_node])
                self.sum_val[node] = self.sum_val[left_node] + self.sum_val[right_node]
            self.lazy_ceiling[node] = INF

    def apply(self, node, l, r, ceiling_val):
        if ceiling_val >= self.max_val[node]:
            return
        if ceiling_val <= self.min_val[node]:
            self.min_val[node] = ceiling_val
            self.max_val[node] = ceiling_val
            self.sum_val[node] = ceiling_val * self.actual_size[node]
            if l != r:
                self.lazy_ceiling[2*node+1] = min(self.lazy_ceiling[2*node+1], ceiling_val)
                self.lazy_ceiling[2*node+2] = min(self.lazy_ceiling[2*node+2], ceiling_val)
            return
        if l == r:
            if ceiling_val < self.min_val[node]:
                self.min_val[node] = ceiling_val
                self.max_val[node] = ceiling_val
                self.sum_val[node] = ceiling_val
            return
        mid = (l + r) // 2
        self.push(node, l, r)
        self.apply(2*node+1, l, mid, ceiling_val)
        self.apply(2*node+2, mid+1, r, ceiling_val)
        self.min_val[node] = min(self.min_val[2*node+1], self.min_val[2*node+2])
        self.max_val[node] = max(self.max_val[2*node+1], self.max_val[2*node+2])
        self.sum_val[node] = self.sum_val[2*node+1] + self.sum_val[2*node+2]

    def range_update_min(self, node, segl, segr, l, r, ceiling_val):
        if segr < l or segl > r:
            return
        if l <= segl and segr <= r:
            self.apply(node, segl, segr, ceiling_val)
            return
        self.push(node, segl, segr)
        mid = (segl + segr) // 2
        if l <= mid:
            self.range_update_min(2*node+1, segl, mid, l, r, ceiling_val)
        if r > mid:
            self.range_update_min(2*node+2, mid+1, segr, l, r, ceiling_val)
        self.min_val[node] = min(self.min_val[2*node+1], self.min_val[2*node+2])
        self.max_val[node] = max(self.max_val[2*node+1], self.max_val[2*node+2])
        self.sum_val[node] = self.sum_val[2*node+1] + self.sum_val[2*node+2]

    def query_threshold(self, node, segl, segr, l, r, T):
        if segr < l or segl > r:
            return (0, 0)
        self.push(node, segl, segr)
        if l <= segl and segr <= r:
            if self.min_val[node] > T:
                return (self.actual_size[node], self.sum_val[node])
            if self.max_val[node] <= T:
                return (0, 0)
        mid = (segl + segr) // 2
        left_res = (0, 0)
        right_res = (0, 0)
        if l <= mid:
            left_res = self.query_threshold(2*node+1, segl, mid, l, r, T)
        if r > mid:
            right_res = self.query_threshold(2*node+2, mid+1, segr, l, r, T)
        return (left_res[0] + right_res[0], left_res[1] + right_res[1])
    
    def point_set(self, node, segl, segr, index, x):
        if segl == segr:
            self.min_val[node] = x
            self.max_val[node] = x
            self.sum_val[node] = x
            self.lazy_ceiling[node] = INF
            return
        self.push(node, segl, segr)
        mid = (segl + segr) // 2
        if index <= mid:
            self.point_set(2*node+1, segl, mid, index, x)
        else:
            self.point_set(2*node+2, mid+1, segr, index, x)
        self.min_val[node] = min(self.min_val[2*node+1], self.min_val[2*node+2])
        self.max_val[node] = max(self.max_val[2*node+1], self.max_val[2*node+2])
        self.sum_val[node] = self.sum_val[2*node+1] + self.sum_val[2*node+2]
    
    def get_point(self, node, segl, segr, index):
        if segl == segr:
            return self.min_val[node]
        self.push(node, segl, segr)
        mid = (segl + segr) // 2
        if index <= mid:
            return self.get_point(2*node+1, segl, mid, index)
        else:
            return self.get_point(2*node+2, mid+1, segr, index)

def main():
    data = sys.stdin.read().split()
    if not data:
        return
    n = int(data[0])
    X = list(map(int, data[1:1+n]))
    q = int(data[1+n])
    tasks = []
    index = 1+n+1
    for i in range(q):
        t = int(data[index]); g = int(data[index+1]); index += 2
        tasks.append((t, g))
    
    A_initial = [X[j] - j for j in range(n)]
    B_initial = [j - X[j] for j in range(n)]
    
    segA = SegmentTree(n, A_initial)
    segB = SegmentTree(n, B_initial)
    
    total_cost = 0
    for (t, g) in tasks:
        i0 = t-1
        c = i0 - g
        if i0 > 0:
            count_left, sum_left = segA.query_threshold(0, 0, n-1, 0, i0-1, -c)
            cost_left = sum_left + c * count_left
        else:
            cost_left = 0
        
        d = g - i0
        if i0 < n-1:
            count_right, sum_right = segB.query_threshold(0, 0, n-1, i0+1, n-1, -d)
            cost_right = sum_right + d * count_right
        else:
            cost_right = 0
            
        a_val = segA.get_point(0, 0, n-1, i0)
        cost_fixed = abs(a_val + i0 - g)
        total_cost += cost_left + cost_right + cost_fixed
        
        segA.point_set(0, 0, n-1, i0, g - i0)
        segB.point_set(0, 0, n-1, i0, i0 - g)
        
        if i0 > 0:
            segA.range_update_min(0, 0, n-1, 0, i0-1, g - i0)
        if i0 < n-1:
            segB.range_update_min(0, 0, n-1, i0+1, n-1, i0 - g)
            
    print(total_cost)

if __name__ == '__main__':
    main()