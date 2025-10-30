class FenwickTree:
    def __init__(self, size):
        self.n = size
        self.tree = [0] * (self.n + 1)  # 1-based indexing

    def update_point(self, idx, delta):
        # Convert to 1-based index
        idx += 1
        while idx <= self.n:
            self.tree[idx] += delta
            idx += idx & -idx

    def query_prefix(self, idx):
        # Sum from 0-based 0 to idx (inclusive)
        # Handle idx < 0 as sum from 0 to -1 (returns 0)
        if idx < 0:
            return 0
        # Convert to 1-based index
        idx += 1
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
    S = input[ptr]
    ptr += 1

    # Initialize array A
    A = []
    for i in range(N-1):
        if S[i] == S[i+1]:
            A.append(1)
        else:
            A.append(0)
    ft = FenwickTree(len(A))
    for i in range(len(A)):
        ft.update_point(i, A[i])

    for _ in range(Q):
        type_q = int(input[ptr])
        ptr += 1
        L = int(input[ptr])
        ptr += 1
        R = int(input[ptr])
        ptr += 1

        if type_q == 1:
            # Flip operation
            if L > 1:
                idx = L - 2
                new_val = 1 - A[idx]
                delta = new_val - A[idx]
                ft.update_point(idx, delta)
                A[idx] = new_val
            if R < N:
                idx = R - 1
                new_val = 1 - A[idx]
                delta = new_val - A[idx]
                ft.update_point(idx, delta)
                A[idx] = new_val
        else:
            # Query operation
            if L == R:
                print("Yes")
            else:
                a = L - 1
                b = R - 2
                sum_val = ft.query_prefix(b) - ft.query_prefix(a - 1)
                print("Yes" if sum_val == 0 else "No")

if __name__ == "__main__":
    main()