class Fenw:
    def __init__(self, size):
        self.n = size
        self.tree = [0] * (self.n + 1)
    
    def update(self, idx, delta):
        pos = idx + 1
        while pos <= self.n:
            self.tree[pos] += delta
            pos += pos & -pos
            
    def query(self, idx):
        if idx < 0:
            return 0
        res = 0
        pos = idx + 1
        while pos > 0:
            res += self.tree[pos]
            pos -= pos & -pos
        return res

def main():
    import sys
    data = sys.stdin.read().split()
    n = int(data[0])
    m = int(data[1])
    A = list(map(int, data[2:2+n]))
    
    S = [0] * (n + 1)
    for i in range(1, n + 1):
        S[i] = (S[i-1] + A[i-1]) % m
        
    part1 = 0
    for i in range(1, n + 1):
        part1 += i * S[i]
    for i in range(0, n):
        part1 -= (n - i) * S[i]
        
    fenw = Fenw(m)
    inv_count = 0
    for j in range(0, n + 1):
        if j > 0:
            cnt_le = fenw.query(S[j])
            cnt_gt = j - cnt_le
            inv_count += cnt_gt
        fenw.update(S[j], 1)
        
    part2 = m * inv_count
    ans = part1 + part2
    print(ans)

if __name__ == "__main__":
    main()