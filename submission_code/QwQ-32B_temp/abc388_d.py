import sys

class FenwickTree:
    def __init__(self, max_size):
        self.size = max_size
        self.tree = [0] * (self.size + 2)  # 1-based indexing

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
    N = int(sys.stdin.readline())
    A = list(map(int, sys.stdin.readline().split()))
    max_val = 1500000  # Maximum possible value of A_i + received_i + i
    ft = FenwickTree(max_val)
    B = []
    for i in range(1, N + 1):
        query_val = i - 1
        current_size = i - 1
        count = current_size - ft.query(query_val)
        received_i = count
        a_i = A[i - 1]
        val_i = a_i + received_i + i
        given_i = min(a_i + received_i, N - i)
        b_i = a_i + received_i - given_i
        B.append(b_i)
        # Ensure val_i is within the Fenwick Tree's size
        if val_i > max_val:
            val_i = max_val
        ft.update(val_i, 1)
    print(' '.join(map(str, B)))

if __name__ == "__main__":
    main()