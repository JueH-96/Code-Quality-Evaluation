import bisect

class SegmentTree:
    def __init__(self, data):
        self.n = len(data)
        self.size = 1
        while self.size < self.n:
            self.size <<= 1
        self.min_val = [float('inf')] * (2 * self.size)
        self.min_idx = [-1] * (2 * self.size)
        for i in range(self.n):
            self.min_val[self.size + i] = data[i]
            self.min_idx[self.size + i] = i
        for i in range(self.size - 1, 0, -1):
            left = 2 * i
            right = 2 * i + 1
            if self.min_val[left] <= self.min_val[right]:
                self.min_val[i] = self.min_val[left]
                self.min_idx[i] = self.min_idx[left]
            else:
                self.min_val[i] = self.min_val[right]
                self.min_idx[i] = self.min_idx[right]
    
    def mark_used(self, idx):
        pos = self.size + idx
        self.min_val[pos] = float('inf')
        self.min_idx[pos] = -1
        pos >>= 1
        while pos >= 1:
            left = 2 * pos
            right = 2 * pos + 1
            if self.min_val[left] <= self.min_val[right]:
                self.min_val[pos] = self.min_val[left]
                self.min_idx[pos] = self.min_idx[left]
            else:
                self.min_val[pos] = self.min_val[right]
                self.min_idx[pos] = self.min_idx[right]
            pos >>= 1
    
    def query_min(self, l, r):
        res_val = float('inf')
        res_idx = -1
        l += self.size
        r += self.size
        while l <= r:
            if l % 2 == 1:
                if self.min_val[l] < res_val:
                    res_val = self.min_val[l]
                    res_idx = self.min_idx[l]
                l += 1
            if r % 2 == 0:
                if self.min_val[r] < res_val:
                    res_val = self.min_val[r]
                    res_idx = self.min_idx[r]
                r -= 1
            l >>= 1
            r >>= 1
        return (res_val, res_idx)

def main():
    import sys
    input = sys.stdin.read().split()
    ptr = 0
    N = int(input[ptr])
    ptr += 1
    M = int(input[ptr])
    ptr +=1
    P = list(map(int, input[ptr:ptr+N]))
    ptr +=N
    L = list(map(int, input[ptr:ptr+M]))
    ptr +=M
    D = list(map(int, input[ptr:ptr+M]))
    ptr +=M
    
    P_sorted = sorted(P)
    coupons = sorted(zip(D, L), key=lambda x: (-x[0], x[1]))
    
    st = SegmentTree(P_sorted)
    
    total_discount = 0
    
    for d, l in coupons:
        pos = bisect.bisect_left(P_sorted, l)
        if pos >= len(P_sorted):
            continue
        min_val, min_idx = st.query_min(pos, len(P_sorted)-1)
        if min_val != float('inf'):
            total_discount += d
            st.mark_used(min_idx)
    
    total = sum(P)
    print(total - total_discount)

if __name__ == '__main__':
    main()