import sys

class FenwickTree:
    def __init__(self, size):
        self.n = size
        self.tree = [0] * (self.n + 1)  # 1-based indexing

    def update(self, index, delta):
        while index <= self.n:
            self.tree[index] += delta
            index += index & -index

    def query(self, index):
        res = 0
        while index > 0:
            res += self.tree[index]
            index -= index & -index
        return res

    def range_query(self, l, r):
        if l > r:
            return 0
        return self.query(r) - self.query(l - 1)

def main():
    input = sys.stdin.read().split()
    ptr = 0
    N = int(input[ptr])
    ptr += 1
    Q = int(input[ptr])
    ptr += 1
    S = input[ptr]
    ptr += 1

    if N == 1:
        current_bad = []
    else:
        current_bad = [0] * (N - 1)
        for i in range(N - 1):
            if S[i] == S[i + 1]:
                current_bad[i] = 1
            else:
                current_bad[i] = 0

    if N > 1:
        ft = FenwickTree(N - 1)
        for i in range(N - 1):
            ft.update(i + 1, current_bad[i])
    else:
        ft = None

    output = []
    for _ in range(Q):
        query_type = input[ptr]
        ptr += 1
        L = int(input[ptr])
        ptr += 1
        R = int(input[ptr])
        ptr += 1

        if query_type == '1':
            # Flip operation
            L0 = L - 1  # 0-based in string
            R0 = R - 1

            # Process L-1
            if L0 > 0:
                i = L0 - 1  # 0-based in current_bad
                if i < len(current_bad):
                    old = current_bad[i]
                    new = 1 - old
                    delta = new - old
                    current_bad[i] = new
                    ft.update(i + 1, delta)  # Fenwick is 1-based

            # Process R
            if R0 < len(S) - 1:
                i = R0  # 0-based in current_bad
                if i < len(current_bad):
                    old = current_bad[i]
                    new = 1 - old
                    delta = new - old
                    current_bad[i] = new
                    ft.update(i + 1, delta)
        else:
            # Check if good
            if L == R:
                output.append("Yes")
                continue

            if N == 1:
                output.append("Yes")
                continue

            a = L - 1  # 0-based in current_bad
            b = R - 2  # 0-based in current_bad

            # Ensure a and b are within bounds
            a = max(0, a)
            b = min(len(current_bad) - 1, b)

            if a > b:
                sum_ = 0
            else:
                sum_ = ft.range_query(a + 1, b + 1)  # Fenwick is 1-based

            if sum_ == 0:
                output.append("Yes")
            else:
                output.append("No")

    print('
'.join(output))

if __name__ == "__main__":
    main()