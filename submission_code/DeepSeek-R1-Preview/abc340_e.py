class FenwickTree:
    def __init__(self, size):
        self.n = size
        self.tree = [0] * (self.n + 2)  # 1-based indexing

    def add(self, idx, val):
        while idx <= self.n + 1:
            self.tree[idx] += val
            idx += idx & -idx

    def query(self, idx):
        res = 0
        while idx > 0:
            res += self.tree[idx]
            idx -= idx & -idx
        return res

    def range_add(self, a, b, val):
        self.add(a, val)
        self.add(b + 1, -val)

def main():
    import sys
    input = sys.stdin.read().split()
    ptr = 0
    N = int(input[ptr])
    ptr += 1
    M = int(input[ptr])
    ptr += 1
    A_initial = list(map(int, input[ptr:ptr + N]))
    ptr += N
    B_list = list(map(int, input[ptr:ptr + M]))
    ptr += M

    global_full = 0
    fenwick = FenwickTree(N)

    for B in B_list:
        # Compute current partial contribution for B
        current_partial = fenwick.query(B + 1)  # B is 0-based, converted to 1-based
        current_value = A_initial[B] + global_full + current_partial
        K = current_value

        # Update A_initial[B] to account for zeroing and current state
        A_initial[B] = -(global_full + current_partial)

        full = K // N
        rem = K % N

        # Update global_full
        global_full += full

        if rem == 0:
            continue

        # Calculate the start and end of the range for rem steps
        S = (B + 1) % N  # 0-based
        end = (S + rem - 1) % N

        # Convert to 1-based indices for Fenwick Tree
        S_1 = S + 1
        end_1 = end + 1

        if S <= end:
            fenwick.range_add(S_1, end_1, 1)
        else:
            # Split into two ranges: [S, N-1] and [0, end]
            fenwick.range_add(S_1, N, 1)
            fenwick.range_add(1, end_1, 1)

    # Compute partials after all operations
    partials = [fenwick.query(x + 1) for x in range(N)]

    # Calculate the final values for each box
    final = [A_initial[x] + global_full + partials[x] for x in range(N)]

    # Output the result
    print(' '.join(map(str, final)))

if __name__ == '__main__':
    main()