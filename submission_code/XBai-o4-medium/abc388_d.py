class FenwickTree:
    def __init__(self, size):
        self.n = size
        self.tree = [0] * (self.n + 2)  # Using n+2 to avoid issues with 1-based indexing

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
    N = int(input[0])
    A = list(map(int, input[1:N+1]))
    max_val = N
    ft = FenwickTree(max_val)
    B = []
    for i in range(1, N+1):
        x = i
        pre_sum = ft.query(x-1)
        current_size = i-1
        R_i = current_size - pre_sum
        S_i = A[i-1] + R_i
        G_i = min(S_i, N - i)
        val = G_i + i
        ft.update(val, 1)
        B_i = A[i-1] + R_i - G_i
        B.append(str(B_i))
    print(' '.join(B))

if __name__ == '__main__':
    main()