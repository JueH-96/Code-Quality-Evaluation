import sys

class FenwickTree:
    def __init__(self, size):
        self.n = size
        self.tree = [0] * (self.n + 1)  # 1-based indexing

    def update(self, index, delta):
        # Add delta to the element at index (1-based)
        while index <= self.n:
            self.tree[index] += delta
            index += index & -index

    def query(self, index):
        # Sum from 1 to index (1-based)
        res = 0
        while index > 0:
            res += self.tree[index]
            index -= index & -index
        return res

    def range_sum(self, l, r):
        # Sum from l to r, inclusive
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
        a = None  # no array
        bit = None
    else:
        a = [0] * (N)  # 1-based, a[1] to a[N-1]
        for i in range(1, N):
            if S[i-1] == S[i]:
                a[i] = 1
            else:
                a[i] = 0
        bit = FenwickTree(N-1)
        for i in range(1, N):
            bit.update(i, a[i])

    output = []
    for _ in range(Q):
        t = input[ptr]
        ptr += 1
        L = int(input[ptr])
        ptr += 1
        R = int(input[ptr])
        ptr += 1
        if t == '1':
            if N != 1:
                # process flip
                if L >= 2:
                    i = L - 1
                    if a[i] == 1:
                        delta = -1
                    else:
                        delta = 1
                    a[i] = 1 - a[i]
                    bit.update(i, delta)
                if R <= (N-1):
                    i = R
                    if a[i] == 1:
                        delta = -1
                    else:
                        delta = 1
                    a[i] = 1 - a[i]
                    bit.update(i, delta)
        else:
            # type 2 query
            if L == R:
                output.append("Yes")
            else:
                l = L
                r = R - 1
                sum_val = bit.range_sum(l, r)
                if sum_val == 0:
                    output.append("Yes")
                else:
                    output.append("No")
    print('
'.join(output))

if __name__ == '__main__':
    main()