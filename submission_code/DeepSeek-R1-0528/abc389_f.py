import sys

class Fenw:
    def __init__(self, size):
        self.n = size
        self.fenw = [0] * (self.n + 1)
    
    def update(self, index, delta):
        while index <= self.n:
            self.fenw[index] += delta
            index += index & -index

    def query(self, index):
        s = 0
        while index:
            s += self.fenw[index]
            index -= index & -index
        return s

def main():
    data = sys.stdin.read().split()
    it = iter(data)
    N = int(next(it))
    contests = []
    for _ in range(N):
        L = int(next(it))
        R = int(next(it))
        contests.append((L, R))
    
    max_val = 500000
    size = max_val + 2
    fenw_tree = Fenw(size)
    
    def query_index(x):
        return fenw_tree.query(x + 1)
    
    def update_index(i, delta):
        fenw_tree.update(i + 1, delta)
    
    def range_add(l, r, delta):
        update_index(l, delta)
        if r + 1 <= max_val:
            update_index(r + 1, -delta)
    
    for L, R in contests:
        low, high = 0, max_val
        a = None
        while low <= high:
            mid = (low + high) // 2
            g_mid = mid + query_index(mid)
            if g_mid >= L:
                a = mid
                high = mid - 1
            else:
                low = mid + 1
        if a is None:
            continue
        
        low, high = 0, max_val
        b = None
        while low <= high:
            mid = (low + high + 1) // 2
            g_mid = mid + query_index(mid)
            if g_mid <= R:
                b = mid
                low = mid + 1
            else:
                high = mid - 1
        if b is None:
            continue
        
        if a <= b:
            range_add(a, b, 1)
    
    Q = int(next(it))
    out_lines = []
    for _ in range(Q):
        x = int(next(it))
        res = x + query_index(x)
        out_lines.append(str(res))
    
    sys.stdout.write("
".join(out_lines))

if __name__ == "__main__":
    main()