class FenwickTree:
    def __init__(self, size):
        self.n = size
        self.tree = [0] * (self.n + 2)  # 1-based indexing

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
    ptr = 0
    N = int(input[ptr])
    ptr += 1
    Q = int(input[ptr])
    ptr += 1
    H = list(map(int, input[ptr:ptr+N]))
    ptr += N
    H = [0] + H  # Convert to 1-based index

    # Compute A_j using a monotonic stack
    A = [0] * (N + 2)  # A[1] to A[N]
    stack = []
    for j in range(1, N + 1):
        while stack and H[stack[-1]] <= H[j]:
            stack.pop()
        if not stack:
            A[j] = 1
        else:
            A[j] = stack[-1] + 1
        stack.append(j)

    # Read and store queries along with their original index
    queries = []
    for idx in range(Q):
        l = int(input[ptr])
        ptr += 1
        r = int(input[ptr])
        ptr += 1
        queries.append((r, l, idx))

    # Sort queries by r in descending order
    sorted_queries = sorted(queries, key=lambda x: (-x[0], x[1]))

    # Initialize Fenwick Tree and answer array
    ft = FenwickTree(N)
    j = N
    ans = [0] * Q

    for r, l, idx in sorted_queries:
        # Insert all buildings j where j > r into the Fenwick Tree
        while j > r:
            if 1 <= j <= N:
                ft.update(A[j], 1)
            j -= 1
        # Query the number of elements <= l
        ans[idx] = ft.query(l)

    # Print answers in the original order
    for a in ans:
        print(a)

if __name__ == '__main__':
    main()