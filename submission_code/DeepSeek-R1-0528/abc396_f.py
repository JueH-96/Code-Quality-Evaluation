import sys

class Fenw:
    def __init__(self, size):
        self.n = size
        self.tree = [0] * (self.n + 1)
    
    def update(self, index, delta):
        pos = index + 1
        while pos <= self.n:
            self.tree[pos] += delta
            pos += pos & -pos
            
    def prefix(self, index):
        if index < 0:
            return 0
        if index >= self.n:
            index = self.n - 1
        pos = index + 1
        s = 0
        while pos:
            s += self.tree[pos]
            pos -= pos & -pos
        return s

def main():
    data = sys.stdin.read().split()
    if not data:
        return
    n = int(data[0])
    m = int(data[1])
    A = list(map(int, data[2:2 + n]))
    
    fenw_val = Fenw(m)
    inv_count = 0
    for i in range(n - 1, -1, -1):
        a = A[i]
        if a > 0:
            cnt = fenw_val.prefix(a - 1)
            inv_count += cnt
        fenw_val.update(a, 1)
    
    fenw_X = Fenw(n)
    fenw_Y = Fenw(n)
    for i in range(n):
        fenw_X.update(i, 1)
    
    by_value = [[] for _ in range(m)]
    for idx, a_val in enumerate(A):
        by_value[a_val].append(idx)
    
    nX = n
    nY = 0
    T1 = 0
    ans = [0] * m
    ans[0] = inv_count
    
    for v in range(m - 1, 0, -1):
        for idx in by_value[v]:
            count1 = nY - fenw_Y.prefix(idx)
            count2 = fenw_X.prefix(idx - 1)
            T1 = T1 - count1 + count2
            fenw_X.update(idx, -1)
            fenw_Y.update(idx, 1)
            nX -= 1
            nY += 1
        
        k_val = m - v
        ans[k_val] = inv_count + 2 * T1 - nX * nY
    
    for k in range(m):
        print(ans[k])

if __name__ == "__main__":
    main()