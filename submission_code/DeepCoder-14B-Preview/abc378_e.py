class FenwickTree:
    def __init__(self, size):
        self.size = size
        self.tree = [0] * (self.size + 1)  # 1-based indexing

    def add(self, index, delta):
        while index <= self.size:
            self.tree[index] += delta
            index += index & -index

    def query(self, index):
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

    # Compute prefix sums
    prefix = [0]
    for a in A:
        prefix.append(prefix[-1] + a)

    # Initialize Fenwick Tree
    fenwick = FenwickTree(M)
    # Initially, S[0] mod M is 0, which is stored at index 1.
    fenwick.add(1, 1)

    sum_s = 0  # sum of s_j for j < r
    total = 0

    for r in range(1, len(prefix)):
        current_sum = prefix[r]
        K = current_sum % M

        # Calculate count_gt
        if K + 1 > M:
            count_gt = 0
        else:
            count_gt = fenwick.query(M) - fenwick.query(K + 1)

        sum_mod = (K * r - sum_s) + M * count_gt
        total += sum_mod

        # Update sum_s and Fenwick Tree
        sum_s += K
        fenwick.add(K + 1, 1)

    print(total)

if __name__ == '__main__':
    main()