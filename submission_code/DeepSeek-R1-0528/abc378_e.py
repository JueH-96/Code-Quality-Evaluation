import sys

class Fenw:
    def __init__(self, size):
        self.n = size
        self.tree = [0] * (self.n + 1)

    def update(self, index, delta):
        i = index + 1
        while i <= self.n:
            self.tree[i] += delta
            i += i & -i

    def query(self, index):
        if index < 0:
            return 0
        res = 0
        i = index + 1
        while i:
            res += self.tree[i]
            i -= i & -i
        return res

def main():
    data = sys.stdin.read().split()
    n = int(data[0])
    M_val = int(data[1])
    A = list(map(int, data[2:2 + n]))
    
    P = [0] * (n + 1)
    for i in range(1, n + 1):
        P[i] = (P[i - 1] + A[i - 1]) % M_val

    part1 = 0
    for i in range(n + 1):
        part1 += P[i] * (2 * i - n)
        
    bit = Fenw(M_val)
    inv_count = 0
    for j in range(n + 1):
        count_le = bit.query(P[j])
        inv_count += j - count_le
        bit.update(P[j], 1)
        
    ans = part1 + M_val * inv_count
    print(ans)

if __name__ == '__main__':
    main()