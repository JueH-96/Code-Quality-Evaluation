import bisect

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    idx = 0
    N = int(data[idx])
    idx += 1
    Q = int(data[idx])
    idx += 1
    
    H = list(map(int, data[idx:idx+N]))
    idx += N
    
    # Compute left array (nearest smaller to the left)
    left = [0] * N
    stack = []
    for i in range(N):
        while stack and H[stack[-1]] >= H[i]:
            stack.pop()
        if stack:
            left[i] = stack[-1]
        else:
            left[i] = -1
        stack.append(i)
    
    # Compute right array (nearest smaller to the right)
    right = [0] * N
    stack = []
    for i in range(N-1, -1, -1):
        while stack and H[stack[-1]] >= H[i]:
            stack.pop()
        if stack:
            right[i] = stack[-1]
        else:
            right[i] = N
        stack.append(i)
    
    # Build a segment tree with sorted lists
    class SegmentTree:
        def __init__(self, data):
            self.n = len(data)
            self.size = 1
            while self.size < self.n:
                self.size <<= 1
            self.tree = [[] for _ in range(2 * self.size)]
            for i in range(self.n):
                self.tree[self.size + i] = [left[i], right[i]]
            for i in range(self.size - 1, 0, -1):
                self.tree[i] = sorted(self.tree[2*i] + self.tree[2*i+1])
        
        def query(self, l, r):
            res = 0
            l += self.size
            r += self.size
            while l <= r:
                if l % 2 == 1:
                    res += self._count(self.tree[l], l, r, 0, self.n-1, -1, float('inf'))
                    l += 1
                if r % 2 == 0:
                    res += self._count(self.tree[r], 0, self.n-1, l, r, float('inf'), -1)
                    r -= 1
                l >>= 1
                r >>= 1
            return res
        
        def _count(self, arr, a, b, l, r, left_val, right_val):
            low = bisect.bisect_left(arr, (left_val, right_val))
            high = bisect.bisect_right(arr, (left_val, right_val))
            return high - low
    
    st = SegmentTree(left)
    
    for _ in range(Q):
        l = int(data[idx])
        idx += 1
        r = int(data[idx])
        idx += 1
        
        a = l - 1
        b = r + 1
        
        if a < 0 or b >= N:
            print(0)
            continue
        
        total = 0
        total += st.query(b, N-1)
        print(total)

if __name__ == '__main__':
    main()