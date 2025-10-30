class BIT:
    def __init__(self, size):
        self.size = size
        self.tree = [0] * (size + 2)  # 1-based indexing

    def update(self, idx, delta=1):
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
    input = sys.stdin.read
    data = input().split()
    ptr = 0
    n = int(data[ptr])
    ptr += 1
    q = int(data[ptr])
    ptr += 1
    H = list(map(int, data[ptr:ptr+n]))
    ptr += n

    N = n
    L = [0] * (N + 1)  # 1-based
    stack = []
    for j in range(1, N + 1):
        while stack and H[stack[-1] - 1] <= H[j - 1]:
            stack.pop()
        if not stack:
            L[j] = 1
        else:
            L[j] = stack[-1] + 1
        stack.append(j)

    queries = []
    ans = [0] * q
    query_idx = 0
    for i in range(q):
        l = int(data[ptr])
        ptr += 1
        r = int(data[ptr])
        ptr += 1
        if r >= N:
            ans[i] = 0
        else:
            queries.append((r, l, i))

    # Sort queries in descending order of r
    queries.sort(reverse=True, key=lambda x: x[0])

    bit = BIT(N)
    current_query = 0
    for j in range(N, 0, -1):
        bit.update(L[j])
        while current_query < len(queries) and queries[current_query][0] == j - 1:
            r, l, idx = queries[current_query]
            res = bit.query(l)
            ans[idx] = res
            current_query += 1

    for a in ans:
        print(a)

if __name__ == "__main__":
    main()