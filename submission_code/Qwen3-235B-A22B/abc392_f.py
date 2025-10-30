import sys

class FenwickTree:
    def __init__(self, size):
        self.n = size
        self.tree = [0] * (self.n + 2)  # Using 1-based indexing

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

def find_kth(bit, k):
    low = 1
    high = bit.n
    while low < high:
        mid = (low + high) // 2
        s = bit.query(mid)
        if s < k:
            low = mid + 1
        else:
            high = mid
    return low

def main():
    input = sys.stdin.read().split()
    n = int(input[0])
    p = list(map(int, input[1:n+1]))

    # Initialize Fenwick Tree with 1s
    bit = FenwickTree(n)
    for i in range(1, n+1):
        bit.update(i, 1)

    pos = [0] * (n + 1)  # pos[i] is the position of element i in the final array

    for i in range(n, 0, -1):
        k = p[i-1]
        x = find_kth(bit, k)
        pos[i] = x
        bit.update(x, -1)

    # Build the final array
    A = [0] * (n + 1)
    for i in range(1, n+1):
        A[pos[i]] = i

    print(' '.join(map(str, A[1:n+1])))

if __name__ == "__main__":
    main()