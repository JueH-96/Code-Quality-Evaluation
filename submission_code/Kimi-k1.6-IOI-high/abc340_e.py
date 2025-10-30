class FenwickTree:
    def __init__(self, size):
        self.n = size
        self.tree = [0] * (self.n + 2)  # Using 1-based indexing

    def add(self, idx, delta):
        idx += 1  # Convert to 1-based index
        while idx <= self.n:
            self.tree[idx] += delta
            idx += idx & -idx

    def query(self, idx):
        idx += 1  # Convert to 1-based index
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
    M = int(data[ptr])
    ptr += 1
    A = list(map(int, data[ptr:ptr+N]))
    ptr += N
    B = list(map(int, data[ptr:ptr+M]))
    ptr += M

    initial_A = A.copy()
    last_total = [0] * N
    fenwick = FenwickTree(N)

    for b in B:
        current_total = fenwick.query(b)
        K = initial_A[b] + (current_total - last_total[b])
        if K == 0:
            continue
        Q, R = divmod(K, N)
        if Q > 0:
            fenwick.add(0, Q)
        if R > 0:
            S = (b + 1) % N
            E = (S + R - 1) % N
            if S <= E:
                fenwick.add(S, 1)
                if E + 1 < N:
                    fenwick.add(E + 1, -1)
            else:
                fenwick.add(S, 1)
                fenwick.add(0, 1)
                if E + 1 < N:
                    fenwick.add(E + 1, -1)
        initial_A[b] = 0
        last_total[b] = current_total

    result = []
    for i in range(N):
        total = initial_A[i] + (fenwick.query(i) - last_total[i])
        result.append(str(total))
    print(' '.join(result))

if __name__ == "__main__":
    main()