import sys

class Fenw:
    def __init__(self, size):
        self.n = size
        self.tree = [0] * (self.n + 1)
    
    def update(self, index, delta):
        while index <= self.n:
            self.tree[index] += delta
            index += index & -index
            
    def query(self, index):
        if index <= 0:
            return 0
        if index > self.n:
            index = self.n
        s = 0
        while index:
            s += self.tree[index]
            index -= index & -index
        return s

def main():
    data = sys.stdin.read().split()
    if not data:
        return
    n = int(data[0])
    A = list(map(int, data[1:1+n]))
    
    MAX_VAL = 2000000
    fenw = Fenw(MAX_VAL)
    g = [0] * n
    
    for i in range(1, n + 1):
        a_val = A[i - 1]
        if i == 1:
            g_i = 0
        else:
            count_le = fenw.query(i - 1)
            g_i = (i - 1) - count_le
        
        v_i = a_val + g_i + i
        if v_i > MAX_VAL:
            v_i = MAX_VAL
        fenw.update(v_i, 1)
        g[i - 1] = g_i
        
    res = []
    for i in range(n):
        total_after = n - (i + 1)
        stones = A[i] + g[i] - total_after
        if stones < 0:
            stones = 0
        res.append(str(stones))
        
    print(" ".join(res))

if __name__ == "__main__":
    main()