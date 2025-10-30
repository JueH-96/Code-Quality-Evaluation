import bisect

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    T = int(data[1])
    S = data[2]
    X = list(map(int, data[3:3+N]))
    
    left_movers = []
    right_movers = []
    for idx in range(N):
        if S[idx] == '0':
            left_movers.append((X[idx], idx + 1))  # (X_j, original index)
        else:
            right_movers.append((X[idx], idx + 1))
    
    left_movers.sort()
    x_list = [x for x, j in left_movers]
    
    if not left_movers:
        print(0)
        return
    
    class MergeSortTree:
        def __init__(self, data):
            self.n = len(data)
            self.size = 1
            while self.size < self.n:
                self.size <<= 1
            self.tree = [[] for _ in range(2 * self.size)]
            for i in range(self.n):
                self.tree[self.size + i] = [data[i][1]]
            for i in range(self.size - 1, 0, -1):
                self.tree[i] = self.merge(self.tree[2 * i], self.tree[2 * i + 1])
        
        @staticmethod
        def merge(a, b):
            res = []
            i = j = 0
            while i < len(a) and j < len(b):
                if a[i] < b[j]:
                    res.append(a[i])
                    i += 1
                else:
                    res.append(b[j])
                    j += 1
            res += a[i:]
            res += b[j:]
            return res
        
        def query(self, l, r, x):
            res = 0
            l += self.size
            r += self.size
            while l < r:
                if l % 2 == 1:
                    res += len(self.tree[l]) - bisect.bisect_right(self.tree[l], x)
                    l += 1
                if r % 2 == 1:
                    r -= 1
                    res += len(self.tree[r]) - bisect.bisect_right(self.tree[r], x)
                l >>= 1
                r >>= 1
            return res
    
    mst = MergeSortTree(left_movers)
    
    ans = 0
    for x_i, i in right_movers:
        upper = x_i + 2 * T
        start = bisect.bisect_right(x_list, x_i)
        end = bisect.bisect_right(x_list, upper)
        if start >= end:
            continue
        ans += mst.query(start, end, i)
    
    print(ans)

if __name__ == '__main__':
    main()