class FenwickTree:
    def __init__(self, size):
        self.size = size
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
    import sys
    input = sys.stdin.read().split()
    ptr = 0
    N = int(input[ptr])
    ptr += 1
    Q = int(input[ptr])
    ptr += 1
    H = list(map(int, input[ptr:ptr+N]))
    ptr += N
    H = [0] + H  # 1-based indexing

    # Compute a_k using a monotonic stack
    a = [0] * (N + 2)  # a[0] and a[N+1] unused
    stack = []
    for k in range(1, N + 1):
        while stack and H[stack[-1]] <= H[k]:
            stack.pop()
        if not stack:
            L = 0
        else:
            L = stack[-1]
        a[k] = L + 1
        stack.append(k)

    # Group queries by their r value
    query_groups = [[] for _ in range(N + 2)]  # r can be up to N
    for idx in range(Q):
        l = int(input[ptr])
        ptr += 1
        r = int(input[ptr])
        ptr += 1
        query_groups[r].append((l, idx))

    # Initialize Fenwick Tree and answer array
    ft = FenwickTree(N)
    ans = [0] * Q

    # Process each building from N down to 1
    for k in range(N, 0, -1):
        ft.update(a[k], 1)
        current_r = k - 1
        for (l, idx) in query_groups[current_r]:
            x = l + 1
            ans[idx] = ft.query(x)

    # Print the answers in the original query order
    for each in ans:
        print(each)

if __name__ == '__main__':
    main()