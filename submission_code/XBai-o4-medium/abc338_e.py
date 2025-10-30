import sys

class FenwickTree:
    def __init__(self, size):
        self.size = size
        self.tree = [0] * (self.size + 1)  # 1-based indexing

    def update(self, idx, delta):
        while idx <= self.size:
            self.tree[idx] += delta
            idx += idx & -idx

    def query(self, idx):
        res = 0
        while idx > 0:
            res += self.tree[idx]
            idx -= idx & -idx
        return res

def main():
    input = sys.stdin.read().split()
    ptr = 0
    N = int(input[ptr])
    ptr += 1
    chords = []
    for _ in range(N):
        a = int(input[ptr])
        b = int(input[ptr+1])
        ptr += 2
        if a > b:
            a, b = b, a
        chords.append((a, b))
    chords.sort()
    max_size = 2 * N
    ft = FenwickTree(max_size)
    for a, b in chords:
        sum1 = ft.query(b - 1)
        sum2 = ft.query(a)
        if sum1 - sum2 > 0:
            print("Yes")
            return
        ft.update(b, 1)
    print("No")

if __name__ == '__main__':
    main()