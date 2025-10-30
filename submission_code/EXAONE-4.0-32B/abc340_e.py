import sys

class FenwTree:
    def __init__(self, size):
        self.n = size
        self.tree = [0] * (self.n + 1)
    
    def update(self, index, delta):
        i = index + 1
        while i <= self.n:
            self.tree[i] += delta
            i += i & -i
            
    def prefix_sum(self, index):
        s = 0
        i = index + 1
        while i > 0:
            s += self.tree[i]
            i -= i & -i
        return s
        
    def range_add(self, l, r, delta):
        self.update(l, delta)
        if r + 1 < self.n:
            self.update(r + 1, -delta)

def main():
    data = sys.stdin.read().split()
    n = int(data[0])
    m = int(data[1])
    A = list(map(int, data[2:2+n]))
    B = list(map(int, data[2+n:2+n+m]))
    
    D = [0] * n
    if n > 0:
        D[0] = A[0]
        for i in range(1, n):
            D[i] = A[i] - A[i-1]
            
    fenw = FenwTree(n)
    for i in range(n):
        fenw.update(i, D[i])
        
    total_base = 0
    
    for b in B:
        val_b = fenw.prefix_sum(b)
        T = val_b + total_base
        
        fenw.range_add(b, b, -T)
        
        base_add = T // n
        rem = T % n
        total_base += base_add
        
        if rem > 0:
            s = (b + 1) % n
            if s + rem <= n:
                fenw.range_add(s, s + rem - 1, 1)
            else:
                fenw.range_add(s, n - 1, 1)
                fenw.range_add(0, s + rem - n - 1, 1)
                
    res = []
    for i in range(n):
        val_i = total_base + fenw.prefix_sum(i)
        res.append(str(val_i))
        
    print(" ".join(res))

if __name__ == "__main__":
    main()