class FenwickTree:
    def __init__(self, size):
        self.N = size
        self.tree = [0] * (self.N + 1)
    
    def update(self, index, delta):
        while index <= self.N:
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
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    Q = int(data[1])
    H = list(map(int, data[2:2+N]))
    queries = []
    for i in range(Q):
        l = int(data[2+N+2*i])
        r = int(data[2+N+2*i+1])
        queries.append((l-1, r-1, i))  # Convert to 0-based indexing
    
    # Compute left_greater[k] for each k using a monotonic stack
    stack = []
    left_greater = [0] * N
    for k in range(N):
        while stack and H[stack[-1]] < H[k]:
            stack.pop()
        if stack:
            left_greater[k] = stack[-1] + 1  # Convert to 1-based indexing
        else:
            left_greater[k] = 0
        stack.append(k)
    
    # Sort queries by r_i in decreasing order
    queries.sort(key=lambda x: x[1], reverse=True)
    
    # Initialize Fenwick tree
    ft = FenwickTree(N + 1)
    
    # Initialize current_k to N-1 (0-based indexing)
    current_k = N - 1
    ans = [0] * Q
    idx = 0
    for l, r, original_idx in queries:
        while current_k > r:
            # Insert left_greater[current_k] + 1 into Fenwick tree
            ft.update(left_greater[current_k] + 1, 1)
            current_k -= 1
        # Query the Fenwick tree for sum from 1 to l + 1
        if l == 0:
            result = ft.query(0)
        else:
            result = ft.query(l + 1)
        ans[original_idx] = result
    
    # Output the answers in the original order
    for a in ans:
        print(a)

if __name__ == "__main__":
    main()