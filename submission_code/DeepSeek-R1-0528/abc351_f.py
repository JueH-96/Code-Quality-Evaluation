import sys

class Fenw:
    __slots__ = ('n', 'tree')
    def __init__(self, n_size):
        self.n = n_size
        self.tree = [0] * (self.n + 1)
    
    def update(self, index, delta):
        i = index + 1
        while i <= self.n:
            self.tree[i] += delta
            i += i & -i
    
    def query(self, index):
        if index < 0:
            return 0
        i = index + 1
        s = 0
        while i:
            s += self.tree[i]
            i -= i & -i
        return s

def main():
    data = sys.stdin.read().split()
    n = int(data[0])
    A = list(map(int, data[1:1 + n]))
    unique_vals = sorted(set(A))
    comp = {val: idx for idx, val in enumerate(unique_vals)}
    size = len(unique_vals)
    bit_count = Fenw(size)
    bit_sum = Fenw(size)
    ans = 0
    for num in A:
        pos = comp[num]
        cnt = bit_count.query(pos - 1)
        total_val = bit_sum.query(pos - 1)
        ans += num * cnt - total_val
        bit_count.update(pos, 1)
        bit_sum.update(pos, num)
    print(ans)

if __name__ == '__main__':
    main()