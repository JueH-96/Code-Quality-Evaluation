import bisect

class FenwickTree:
    def __init__(self, size):
        self.n = size
        self.tree = [0] * (self.n + 2)  # 1-based indexing

    def add(self, idx, delta):
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
    T = int(input[ptr])
    ptr += 1
    S = input[ptr]
    ptr += 1
    X = list(map(int, input[ptr:ptr + N]))
    ptr += N

    # Coordinate compression
    sorted_x = sorted(list(set(X)))
    M = len(sorted_x)
    L_bit = FenwickTree(M)
    R_bit = FenwickTree(M)
    total = 0
    T_plus = T + 0.1

    for i in reversed(range(N)):
        current_X = X[i]
        current_dir = S[i]
        if current_dir == '1':
            a = current_X
            b = current_X + 2 * T_plus
        else:
            a = current_X - 2 * T_plus
            b = current_X

        left = bisect.bisect_left(sorted_x, a)
        right = bisect.bisect_right(sorted_x, b)

        if current_dir == '1':
            # Query L_bit
            cnt = L_bit.query(right) - L_bit.query(left)
        else:
            # Query R_bit
            cnt = R_bit.query(right) - R_bit.query(left)
        total += cnt

        pos = bisect.bisect_left(sorted_x, current_X)
        if current_dir == '1':
            R_bit.add(pos + 1, 1)
        else:
            L_bit.add(pos + 1, 1)

    print(total)

if __name__ == '__main__':
    main()