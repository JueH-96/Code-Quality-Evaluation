class FenwickTree:
    def __init__(self, size):
        self.n = size
        self.tree = [0] * (self.n + 2)  # 1-based indexing
    
    def update(self, idx, delta):
        while idx <= self.n:
            self.tree[idx] += delta
            idx += idx & -idx
    
    def query(self, idx):
        res = 0
        while idx > 0:
            res += self.tree[idx]
            idx -= idx & -idx
        return res

def main():
    import sys
    input = sys.stdin.read().split()
    ptr = 0
    N = int(input[ptr])
    ptr += 1
    points = list(map(int, input[ptr:ptr+2*N]))
    ptr += 2*N
    
    chords = []
    for i in range(N):
        x = points[2*i]
        y = points[2*i + 1]
        d = (y - x) % (2 * N)
        if d < N:
            a = x
            b = y
        else:
            a = y
            b = x
        chords.append((a, b))
    
    chords.sort(key=lambda x: x[0])
    
    max_b = chords[-1][1] if chords else 0
    fenwick = FenwickTree(max_b)
    
    for a, b in chords:
        low = a + 1
        high = b - 1
        if low > high:
            sum_val = 0
        else:
            sum_val = fenwick.query(high) - fenwick.query(low - 1)
        if sum_val > 0:
            print("Yes")
            return
        fenwick.update(b, 1)
    
    print("No")

if __name__ == "__main__":
    main()