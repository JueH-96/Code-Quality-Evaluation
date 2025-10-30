class FenwickTree:
    def __init__(self, size):
        self.size = size
        self.tree = [0] * (self.size + 2)  # 1-based indexing

    def update(self, index, delta):
        # Convert to 1-based index
        index += 1
        while index <= self.size + 1:
            self.tree[index] += delta
            index += index & -index

    def query(self, index):
        # Convert to 1-based index, sum from 0 to index (inclusive)
        index += 1
        res = 0
        while index > 0:
            res += self.tree[index]
            index -= index & -index
        return res

def main():
    import sys
    input = sys.stdin.read().split()
    ptr = 0
    N = int(input[ptr])
    ptr += 1
    M = int(input[ptr])
    ptr += 1
    A = list(map(int, input[ptr:ptr+N]))
    ptr += N

    # Compute prefix mod
    P = [0] * (N + 1)
    for i in range(1, N + 1):
        P[i] = (P[i-1] + A[i-1]) % M

    # Initialize Fenwick Tree
    fenwick = FenwickTree(M)
    fenwick.update(P[0], 1)
    sum_prev = P[0]
    total = 0

    for r in range(1, N + 1):
        Q = P[r]
        cnt = r - fenwick.query(Q)
        sum_mod = Q * r - sum_prev + M * cnt
        total += sum_mod
        fenwick.update(Q, 1)
        sum_prev += Q

    print(total)

if __name__ == '__main__':
    main()