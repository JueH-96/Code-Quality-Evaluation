import sys

class FenwickTree:
    def __init__(self, size):
        self.n = size
        self.tree = [0] * (self.n + 1)  # 1-based indexing

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
    input = sys.stdin.read().split()
    idx = 0
    N = int(input[idx])
    idx += 1
    P = list(map(int, input[idx:idx + N]))
    
    ft = FenwickTree(N)
    for i in range(1, N + 1):
        ft.update(i, 1)
    
    A = [0] * N
    
    for i in range(N, 0, -1):
        k = P[i-1]
        low = 1
        high = N
        res = 0
        while low <= high:
            mid = (low + high) // 2
            s = ft.query(mid)
            if s >= k:
                res = mid
                high = mid - 1
            else:
                low = mid + 1
        A[res - 1] = i
        ft.update(res, -1)
    
    print(' '.join(map(str, A)))

if __name__ == '__main__':
    main()