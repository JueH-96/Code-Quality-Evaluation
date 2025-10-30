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
    input = sys.stdin.read
    data = input().split()
    ptr = 0
    N = int(data[ptr])
    ptr += 1
    Q = int(data[ptr])
    ptr += 1
    S = data[ptr]
    ptr += 1
    
    # Initialize D array (1-based to N-1)
    D = [0] * (N)  # D[1] to D[N-1] are used
    for i in range(1, N):
        if S[i-1] != S[i]:
            D[i] = 1
        else:
            D[i] = 0
    
    # Initialize Fenwick Tree
    if N == 1:
        fenwick = None
    else:
        fenwick = FenwickTree(N-1)
        for i in range(1, N):
            fenwick.update(i, D[i])
    
    output = []
    for _ in range(Q):
        query_type = data[ptr]
        ptr += 1
        L = int(data[ptr])
        ptr += 1
        R = int(data[ptr])
        ptr += 1
        
        if query_type == '1':
            # Flip query
            if N == 1:
                continue  # No effect
            # Toggle D[L-1] if L > 1
            if L > 1:
                pos = L - 1
                if pos <= N - 1:
                    current = D[pos]
                    new_val = 1 - current
                    delta = new_val - current
                    D[pos] = new_val
                    fenwick.update(pos, delta)
            # Toggle D[R] if R < N
            if R < N:
                pos = R
                if pos >= 1 and pos <= N - 1:
                    current = D[pos]
                    new_val = 1 - current
                    delta = new_val - current
                    D[pos] = new_val
                    fenwick.update(pos, delta)
        else:
            # Check query
            if L == R:
                output.append("Yes")
            else:
                if N == 1:
                    output.append("Yes")
                else:
                    left = L
                    right = R - 1
                    total = R - L
                    sum_val = fenwick.query(right) - fenwick.query(left - 1)
                    if sum_val == total:
                        output.append("Yes")
                    else:
                        output.append("No")
    print('
'.join(output))

if __name__ == "__main__":
    main()